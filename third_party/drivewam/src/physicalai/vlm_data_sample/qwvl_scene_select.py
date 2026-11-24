"""
Data-parallel scene filter: tag AV driving clips with Qwen3-VL using 4 themed prompts.

Spawns N worker processes, one Qwen3-VL model per GPU, each processing a
disjoint round-robin shard of the clip list. Per-clip JSON files are written
atomically (tmp → rename) to a shared `output_dir`, so multiple ranks writing
disjoint clip_ids is race-free. Resume works automatically — each worker skips
clip_ids whose JSON already exists in output_dir at startup.

For each clip:
  1. Uniformly sample N frames from the ~20s MP4.
  2. Run 4 themed prompts on the SAME frames (video encoded once, reused):
       - scene_attrs:           weather / lighting / road_type / traffic / ego
       - vru_events:            pedestrian / cyclist interactions
       - vehicle_events:        cut-in / cut-out / sudden brake / etc.
       - intersection_longtail: unprotected_left / accident / debris / etc.
  3. Parse each JSON block, merge into one record.
  4. Compute a rule-based interest_score from the tags.
  5. Append the record to per-clip JSON files (flush per clip, resume-safe).

"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import AutoProcessor
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as Qwen3VLForConditionalGeneration


log = logging.getLogger("qwvl_scene_select")
log.setLevel(logging.INFO)
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
log.addHandler(_h)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYS_BASE = (
    "你是自动驾驶数据标注专家。根据前视行车视频判断场景标签。"
    "严格只输出 JSON，不要任何解释、markdown 标记或其他文字。"
    "事件型字段默认 false，只有视频中**明确发生**、**对自车决策有影响**时才标 true。"
)

P1_SCENE_ATTRS = {
    "name": "scene_attrs",
    "system": _SYS_BASE,
    "user": """请判断以下场景属性，严格按 JSON schema 输出（字段值只能从给出的枚举里选）：
{
  "weather": "clear | rainy | snowy | foggy | unknown",
  "lighting": "day | dusk | night | tunnel_transition | strong_backlight",
  "road_type": "urban | highway | ramp | rural | parking_lot | intersection | roundabout | construction_zone | tunnel",
  "traffic_density": "empty | low | medium | high | congested",
  "ego_behavior": "lane_keeping | lane_change | overtake | unprotected_left | u_turn | merge | emergency_brake | stop_and_go | park"
}
只输出 JSON。""",
}

P2_VRU = {
    "name": "vru_events",
    "system": _SYS_BASE,
    "user": """判断以下 VRU（弱势交通参与者）事件是否发生，默认全部 false。
- pedestrian_crossing: 行人正在或即将横穿自车路径
- pedestrian_jaywalking: 行人闯红灯 / 非斑马线横穿
- occluded_pedestrian_popout: 鬼探头（从遮挡物后突然出现的行人）
- child_or_elderly: 画面中有儿童 / 老人 / 轮椅 / 拄拐
- cyclist_or_ebike_conflict: 自行车或电动车与自车有交互
- crowd: 行人群聚集（学校 / 商圈 / 地铁口等）

严格按 JSON 输出（所有字段值均为 true 或 false）：
{
  "pedestrian_crossing": false,
  "pedestrian_jaywalking": false,
  "occluded_pedestrian_popout": false,
  "child_or_elderly": false,
  "cyclist_or_ebike_conflict": false,
  "crowd": false
}
只输出 JSON。""",
}

P3_VEHICLE = {
    "name": "vehicle_events",
    "system": _SYS_BASE,
    "user": """判断以下车辆事件是否发生，默认全部 false。
- cut_in: 他车近距离加塞进入自车车道
- cut_out: 前车切出让位
- sudden_brake_ahead: 前车突然急刹
- wrong_way_vehicle: 逆行车辆
- large_vehicle_occlusion: 卡车 / 公交 / 挂车遮挡视野
- emergency_vehicle: 救护车 / 消防车 / 警车
- door_opening: 停放车辆突然开门
- stopped_or_broken_vehicle: 路边停放 / 故障车需借道

严格按 JSON 输出（所有字段值均为 true 或 false）：
{
  "cut_in": false,
  "cut_out": false,
  "sudden_brake_ahead": false,
  "wrong_way_vehicle": false,
  "large_vehicle_occlusion": false,
  "emergency_vehicle": false,
  "door_opening": false,
  "stopped_or_broken_vehicle": false
}
只输出 JSON。""",
}

P4_INTERSECTION_LONGTAIL = {
    "name": "intersection_longtail",
    "system": _SYS_BASE,
    "user": """判断以下路口和长尾事件。traffic_light 从枚举选一个；其余字段默认 false，明确发生才 true。
- traffic_light: 自车前方信号灯状态，取 "none | red | yellow | green | flashing | blocked"
- unprotected_left: 无保护左转
- roundabout: 环岛
- irregular_intersection: 五岔 / 六岔 / 不规则路口
- traffic_police_gesture: 交警手势指挥
- road_debris: 抛洒物（纸箱、轮胎、掉落物）
- accident_scene: 事故现场
- construction: 施工区（锥桶、围挡、临时标志）
- animal_on_road: 路面有动物
- water_puddle: 积水
- railway_crossing: 铁路道口

严格按 JSON 输出：
{
  "traffic_light": "none",
  "unprotected_left": false,
  "roundabout": false,
  "irregular_intersection": false,
  "traffic_police_gesture": false,
  "road_debris": false,
  "accident_scene": false,
  "construction": false,
  "animal_on_road": false,
  "water_puddle": false,
  "railway_crossing": false
}
只输出 JSON。""",
}

PROMPTS = [P1_SCENE_ATTRS, P2_VRU, P3_VEHICLE, P4_INTERSECTION_LONGTAIL]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

BOOL_WEIGHTS: Dict[str, float] = {
    # Long-tail / rare
    "accident_scene": 5.0,
    "animal_on_road": 3.5,
    "road_debris": 2.5,
    "railway_crossing": 3.0,
    "traffic_police_gesture": 3.0,
    "water_puddle": 1.5,
    # VRU
    "occluded_pedestrian_popout": 4.0,
    "pedestrian_jaywalking": 3.0,
    "pedestrian_crossing": 2.0,
    "child_or_elderly": 1.0,
    "cyclist_or_ebike_conflict": 2.0,
    "crowd": 1.5,
    # Vehicle interactions
    "cut_in": 2.5,
    "cut_out": 1.0,
    "sudden_brake_ahead": 2.5,
    "wrong_way_vehicle": 3.0,
    "emergency_vehicle": 2.5,
    "door_opening": 3.0,
    "large_vehicle_occlusion": 1.0,
    "stopped_or_broken_vehicle": 1.5,
    # Intersections / structure
    "unprotected_left": 2.0,
    "roundabout": 1.5,
    "irregular_intersection": 2.0,
    "construction": 1.5,
}

ENUM_WEIGHTS: Dict[str, Dict[str, float]] = {
    "weather": {"rainy": 1.5, "snowy": 2.0, "foggy": 2.0},
    "lighting": {"night": 1.0, "strong_backlight": 1.0, "tunnel_transition": 1.5, "dusk": 0.5},
    "traffic_density": {"high": 1.0, "congested": 1.5},
    "traffic_light": {"red": 0.5, "yellow": 1.0, "flashing": 1.0, "blocked": 1.5},
    "ego_behavior": {
        "unprotected_left": 1.5,
        "u_turn": 1.5,
        "overtake": 1.0,
        "emergency_brake": 2.0,
        "stop_and_go": 0.5,
    },
}


def compute_interest_score(record: Dict) -> float:
    score = 0.0
    for sub in record.values():
        if not isinstance(sub, dict):
            continue
        for key, val in sub.items():
            if val is True and key in BOOL_WEIGHTS:
                score += BOOL_WEIGHTS[key]
            elif isinstance(val, str) and key in ENUM_WEIGHTS:
                score += ENUM_WEIGHTS[key].get(val, 0.0)
    return round(score, 3)


def flatten_tags(record: Dict) -> List[str]:
    tags: List[str] = []
    for sub in record.values():
        if not isinstance(sub, dict):
            continue
        for key, val in sub.items():
            if val is True:
                tags.append(key)
            elif isinstance(val, str) and val not in ("none", "unknown", "clear", "day"):
                tags.append(f"{key}:{val}")
    return tags


# ---------------------------------------------------------------------------
# JSON parsing (lenient)
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_block(text: str) -> Dict:
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return {"_error": "no json block", "_raw": text}
    block = m.group(0)
    try:
        return json.loads(block)
    except json.JSONDecodeError as e:
        stripped = re.sub(r"//[^\n]*", "", block)
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return {"_error": str(e), "_raw": text}


# ---------------------------------------------------------------------------
# Video frame loading
# ---------------------------------------------------------------------------

def load_video_frames_rgb(video_path: str) -> tuple[List[np.ndarray], float]:
    """Decode video_path → (frames_rgb, native_fps)."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, float(native_fps)


def sample_uniform(frames: List[np.ndarray], n: int) -> List[np.ndarray]:
    total = len(frames)
    if total == 0:
        return []
    if total <= n:
        return list(frames)
    idxs = np.linspace(0, total - 1, n, dtype=int).tolist()
    return [frames[i] for i in idxs]


# ---------------------------------------------------------------------------
# Clip selection & resume
# ---------------------------------------------------------------------------

def select_clips(
    clip_index_path: str,
    dataset_root: str,
    split: str,
) -> List[str]:
    """Deterministic sorted list of all valid clip_ids in the given split."""
    df = pd.read_parquet(clip_index_path)
    valid = df.loc[(df["split"] == split) & df["clip_is_valid"]]
    existing = set(os.listdir(dataset_root))
    candidates = sorted(str(c) for c in valid.index if c in existing)
    if not candidates:
        raise RuntimeError(f"No valid {split} clips under {dataset_root}")
    return candidates


def load_done_from_dir(output_dir: Path) -> set:
    if not output_dir.exists():
        return set()
    return {p.stem for p in output_dir.glob("*.json")}


# ---------------------------------------------------------------------------
# Per-clip CPU prep
# ---------------------------------------------------------------------------

IMPORTANT_TAGS = {
    "accident_scene", "animal_on_road", "road_debris", "railway_crossing",
    "traffic_police_gesture", "occluded_pedestrian_popout",
    "pedestrian_jaywalking", "wrong_way_vehicle", "emergency_vehicle",
    "door_opening", "cut_in", "sudden_brake_ahead", "unprotected_left",
}


def prepare_clip(
    clip_id: str,
    dataset_root: str,
    num_sample_frames: int,
) -> Optional[Dict]:
    """CPU-side prep: decode, sample frames, compute VLM fps. None on failure."""
    video_path = os.path.join(dataset_root, clip_id, "camera_front_wide_120fov.mp4")
    if not os.path.exists(video_path):
        log.warning(f"[{clip_id}] missing video")
        return None

    try:
        frames, native_fps = load_video_frames_rgb(video_path)
    except Exception:
        log.error(f"[{clip_id}] decode failed:\n{traceback.format_exc()}")
        return None
    if not frames:
        log.warning(f"[{clip_id}] empty clip")
        return None

    sampled = sample_uniform(frames, n=num_sample_frames)
    clip_duration_s = len(frames) / native_fps
    fps_for_vlm = len(sampled) / clip_duration_s

    return {
        "clip_id": clip_id,
        "frames": sampled,
        "fps_for_vlm": fps_for_vlm,
        "num_frames_total": len(frames),
        "clip_duration_s": clip_duration_s,
    }


# ---------------------------------------------------------------------------
# SceneTagger
# ---------------------------------------------------------------------------

class SceneTagger:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 300,
    ):
        log.info(f"Loading Qwen3-VL from {model_path} (device={device}) ...")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            attn_implementation=attn_implementation,
        ).eval()
        self.input_device = self.model.device
        self.max_new_tokens = max_new_tokens
        self.processor.tokenizer.padding_side = "left"
        log.info(f"Qwen3-VL loaded. input_device={self.input_device}")


# ---------------------------------------------------------------------------
# CachedSceneTagger: encode video tensors once per batch, reuse across prompts
# ---------------------------------------------------------------------------
#
# The four prompts in PROMPTS differ only in text. The expensive part of the
# processor — image_processor doing resize + normalize + patchify on every
# video frame — is identical across prompts. CachedSceneTagger calls the
# processor fully once, stashes `pixel_values_videos` / `video_grid_thw`, and
# on subsequent prompts only re-tokenizes text (fast) and splices in the cached
# vision tensors.
# ---------------------------------------------------------------------------


def _collapse_video_kwargs(video_kwargs: Dict) -> Dict:
    """process_vision_info returns list-valued kwargs (one entry per video),
    but the processor validates them as scalars. Collapse to scalar via median."""
    for k, v in list(video_kwargs.items()):
        if isinstance(v, list):
            if len(v) == 0:
                video_kwargs.pop(k)
                continue
            if all(isinstance(x, (int, float)) for x in v):
                lo, hi = min(v), max(v)
                if hi - lo > 1e-3 * max(abs(hi), 1.0):
                    log.warning(
                        f"video_kwargs[{k!r}] varies across batch "
                        f"(min={lo:.3f}, max={hi:.3f}); using median"
                    )
                video_kwargs[k] = float(sorted(v)[len(v) // 2])
            else:
                video_kwargs[k] = v[0]
    return video_kwargs


class CachedSceneTagger(SceneTagger):
    """SceneTagger subclass that reuses visual tensors across prompts on the same batch."""

    @torch.inference_mode()
    def run_all_prompts(
        self,
        batch_frames: List[List[np.ndarray]],
        prompts: List[Dict],
        batch_fps: List[float],
    ) -> List[List[str]]:
        """Run every prompt on the same batch of frames, encoding videos once.

        Returns raws[prompt_idx][clip_idx]."""
        if not batch_frames:
            return [[] for _ in prompts]

        pil_batches = [[Image.fromarray(f) for f in frames] for frames in batch_frames]

        # ----- Prompt 0: full processor call. Stash vision tensors. -----
        texts0, video_inputs, video_kwargs = self._build_text_and_vision(
            pil_batches, prompts[0]["system"], prompts[0]["user"], batch_fps,
        )
        inputs0 = self.processor(
            text=texts0,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.input_device)

        cached: Dict[str, torch.Tensor] = {}
        for k in (
            "pixel_values_videos", "video_grid_thw",
            "pixel_values", "image_grid_thw",
            "second_per_grid_ts",
        ):
            if k in inputs0:
                cached[k] = inputs0[k]

        raws0 = self._generate_and_decode(inputs0)
        all_raws: List[List[str]] = [raws0]

        # ----- Prompts 1..N: re-tokenize text only, splice in cached video. -----
        video_token = getattr(self.processor, "video_token", "<|video_pad|>")
        merge_size = getattr(self.processor.image_processor, "merge_size", 2)
        merge_length = merge_size ** 2
        grid_thw_cpu = cached["video_grid_thw"].cpu() if "video_grid_thw" in cached else None

        for prompt in prompts[1:]:
            texts = self._apply_chat_template(
                pil_batches, prompt["system"], prompt["user"], batch_fps,
            )

            if grid_thw_cpu is not None:
                expanded = []
                for i, t in enumerate(texts):
                    count = int(grid_thw_cpu[i].prod().item()) // merge_length
                    parts = t.split(video_token, 1)
                    if len(parts) == 2:
                        t = parts[0] + (video_token * count) + parts[1]
                    expanded.append(t)
                texts = expanded

            text_inputs = self.processor.tokenizer(
                texts, padding=True, return_tensors="pt",
            )
            text_inputs = {k: v.to(self.input_device) for k, v in text_inputs.items()}

            combined = {**text_inputs, **cached}
            raws = self._generate_and_decode(combined)
            all_raws.append(raws)

        return all_raws

    # --- helpers ---

    def _apply_chat_template(self, pil_batches, system_prompt, user_prompt, batch_fps):
        all_messages = [
            [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": pil, "fps": fps},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
            for pil, fps in zip(pil_batches, batch_fps)
        ]
        return [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for m in all_messages
        ]

    def _build_text_and_vision(self, pil_batches, system_prompt, user_prompt, batch_fps):
        all_messages = [
            [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": pil, "fps": fps},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
            for pil, fps in zip(pil_batches, batch_fps)
        ]
        texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for m in all_messages
        ]
        pv_out = process_vision_info(all_messages, return_video_kwargs=True)
        if isinstance(pv_out, tuple) and len(pv_out) == 3:
            _, video_inputs, video_kwargs = pv_out
        else:
            _, video_inputs = pv_out
            video_kwargs = {}
        video_kwargs = _collapse_video_kwargs(video_kwargs)
        return texts, video_inputs, video_kwargs

    def _generate_and_decode(self, inputs):
        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        input_len = inputs["input_ids"].shape[1]
        new_ids = gen_ids[:, input_len:]
        decoded = self.processor.batch_decode(new_ids, skip_special_tokens=True)
        return [s.strip() for s in decoded]


# ---------------------------------------------------------------------------
# Dataset / collate
# ---------------------------------------------------------------------------

class _ClipDataset(Dataset):
    """CPU-only prep: decode mp4 + uniform-sample frames. __getitem__ returns
    None on failure; the collate_fn drops those out of the batch."""

    def __init__(self, clip_ids: List[str], dataset_root: str, num_sample_frames: int):
        self.clip_ids = clip_ids
        self.dataset_root = dataset_root
        self.num_sample_frames = num_sample_frames

    def __len__(self) -> int:
        return len(self.clip_ids)

    def __getitem__(self, idx: int):
        return prepare_clip(self.clip_ids[idx], self.dataset_root, self.num_sample_frames)


def _collate(batch):
    """Drop Nones from failed prepare_clip() calls. Return (raw_size, valid_list)
    so the worker can advance the progress bar even when some clips were dropped."""
    return len(batch), [b for b in batch if b is not None]


# ---------------------------------------------------------------------------
# Batched tagging
# ---------------------------------------------------------------------------

def _save_record(rec: Dict, output_dir: Path) -> None:
    """Atomic tmp → rename; safe for concurrent writers on disjoint clip_ids."""
    out_path = output_dir / f"{rec['clip_id']}.json"
    tmp_path = out_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, out_path)


def tag_batch_cached(tagger: CachedSceneTagger, clip_data: List[Dict]) -> List[Dict]:
    """Run all 4 prompts on a batch of pre-loaded clips, encoding videos once."""
    records: List[Dict] = [
        {
            "clip_id": c["clip_id"],
            "num_frames_sampled": len(c["frames"]),
            "num_frames_total": c["num_frames_total"],
            "clip_duration_s": round(c["clip_duration_s"], 3),
            "fps_for_vlm": round(c["fps_for_vlm"], 3),
        }
        for c in clip_data
    ]
    batch_frames = [c["frames"] for c in clip_data]
    batch_fps = [c["fps_for_vlm"] for c in clip_data]

    try:
        all_raws = tagger.run_all_prompts(
            batch_frames=batch_frames,
            prompts=PROMPTS,
            batch_fps=batch_fps,
        )
    except Exception as e:
        log.error(
            f"[batch size={len(clip_data)}] cached run_all_prompts failed: "
            f"{type(e).__name__}: {e}"
        )
        for rec in records:
            for prompt in PROMPTS:
                rec[prompt["name"]] = {"_error": f"{type(e).__name__}: {e}"}
            rec["interest_score"] = 0.0
            rec["tags"] = []
        return records

    for prompt, raws in zip(PROMPTS, all_raws):
        for rec, raw in zip(records, raws):
            rec[prompt["name"]] = parse_json_block(raw)

    for rec in records:
        rec["interest_score"] = compute_interest_score(rec)
        rec["tags"] = flatten_tags(rec)
    return records


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker_fn(rank: int, args: argparse.Namespace, all_clip_ids: List[str]) -> None:
    device = f"cuda:{rank}"

    # Round-robin shard of the window-sliced clip list
    shard = all_clip_ids[rank :: args.num_gpus]

    output_dir = Path(args.output_dir)
    # Resume. Safe across ranks because shards are disjoint.
    done = load_done_from_dir(output_dir)
    shard = [c for c in shard if c not in done]

    log.info(
        f"[rank {rank}] gpu={rank} shard_size={len(shard)} "
        f"(globally {len(done)} clips already done)"
    )
    if not shard:
        log.info(f"[rank {rank}] nothing to do.")
        return

    # Optional stagger so N ranks don't all hit NFS with 16GB weight reads at once.
    if args.load_stagger_s > 0.0:
        time.sleep(rank * args.load_stagger_s)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    tagger = CachedSceneTagger(
        model_path=args.model_path,
        device=device,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.max_new_tokens,
    )

    # DataLoader overlaps CPU video decoding with GPU inference.
    # multiprocessing_context="spawn" avoids fork-vs-CUDA issues inside mp.spawn'd workers.
    loader = DataLoader(
        _ClipDataset(shard, args.dataset_root, args.num_sample_frames),
        batch_size=args.batch_size,
        num_workers=args.loader_workers,
        collate_fn=_collate,
        shuffle=False,
        pin_memory=False,
        persistent_workers=args.loader_workers > 0,
        prefetch_factor=2 if args.loader_workers > 0 else None,
        multiprocessing_context="spawn" if args.loader_workers > 0 else None,
    )

    pbar = tqdm(total=len(shard), position=rank, leave=True,
                desc=f"GPU{rank}", unit="clip")
    try:
        for raw_n, prepared in loader:
            pbar.update(raw_n)
            if not prepared:
                continue

            records = tag_batch_cached(tagger, prepared)
            for rec in records:
                _save_record(rec, output_dir)
                hit = [t for t in rec["tags"] if t in IMPORTANT_TAGS]
                if hit:
                    log.info(
                        f"[rank {rank}][HIT] clip_id={rec['clip_id']} "
                        f"score={rec['interest_score']} tags={hit}"
                    )
    finally:
        pbar.close()
    log.info(f"[rank {rank}] done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Data-parallel scene filter via Qwen3-VL "
                    "(N models × N GPUs, one process per GPU)"
    )
    parser.add_argument("--model_path",
        default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--dataset_root",
        default="./data/physicalai/front")
    parser.add_argument("--clip_index",
        default="./data/physicalai/clip_index.parquet")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir",
        default=None,
        help="Directory for per-clip JSON files. Shared across ranks (writes are atomic).")
    parser.add_argument("--start_idx", type=int, default=0,
        help="Inclusive start index into the sorted clip list (applied BEFORE round-robin sharding).")
    parser.add_argument("--end_idx", type=int, default=-1,
        help="Exclusive end index. -1 = end of split.")
    parser.add_argument("--num_sample_frames", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4,
        help="Clips per model.generate call (per GPU). Bump up if GPU has headroom.")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--attn_implementation", default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--num_gpus", type=int, default=1,
        help="Number of GPU workers (uses cuda:0 .. cuda:num_gpus-1).")
    parser.add_argument("--load_stagger_s", type=float, default=0.0,
        help="Sleep rank*S seconds before model load to avoid NFS thrash when "
             "many ranks read 16GB of weights simultaneously. 0 = no stagger.")
    parser.add_argument("--loader_workers", type=int, default=12,
        help="DataLoader worker processes (per GPU rank) that decode video in "
             "parallel with GPU inference. 0 = inline decode (no overlap).")
    args = parser.parse_args()

    available = torch.cuda.device_count()
    if args.num_gpus > available:
        raise RuntimeError(f"Requested {args.num_gpus} GPUs, only {available} visible.")

    # Select + window-slice clips in the parent (cheap; a few hundred ms).
    all_clip_ids = select_clips(args.clip_index, args.dataset_root, args.split)
    total = len(all_clip_ids)
    start = max(0, args.start_idx)
    end = total if args.end_idx < 0 else min(total, args.end_idx)
    sliced = all_clip_ids[start:end]
    log.info(
        f"split={args.split} total={total}; window=[{start}:{end}) -> "
        f"{len(sliced)} clips distributed across {args.num_gpus} GPU workers"
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    mp.spawn(_worker_fn, args=(args, sliced), nprocs=args.num_gpus, join=True)

    log.info(f"All ranks finished. Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
