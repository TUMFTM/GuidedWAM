"""
Step 2 of the leak-free per-chunk navigation pipeline.

Generates a navigation-guidance caption for each UPCOMING chunk using ONLY
past observations + a route command — never any frame / BEV / speed from the
upcoming chunk itself.

Inputs per upcoming chunk i of clip c:
  * route_store[chunk_key(c, i)]  (from generate_route.py) — 'straight'|'left'|'right'
  * chunk i == 0: the first front-camera frame of the clip, no BEV history
  * chunk i >= 1: first frame of chunk_i + BEV of chunk_{i-1}

Output JSON is nested: {clip_id: {chunk_idx: guidance_text}}
"""
from __future__ import annotations

import argparse
import os
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image as PILImage

from physicalai.vlm_preprocess.generate_bev import _chunk_key
from physicalai.vlm_preprocess.video_utils import clip_video_path, load_all_frames_rgb
from physicalai.vlm_preprocess.vlm_captioner import VLLMNavCaptioner

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Clip selection
# ----------------------------------------------------------------------------

def select_demo_clips(
    clip_index_path: str,
    dataset_root: str,
    num_clips: int,
    seed: int = 0,
) -> list:
    df = pd.read_parquet(clip_index_path)
    valid = df.loc[(df["split"] == "train") & df["clip_is_valid"]]
    existing = set(os.listdir(dataset_root))
    candidates = [cid for cid in valid.index if cid in existing]
    if not candidates:
        raise RuntimeError(f"No valid train clips found under {dataset_root}")
    rng = np.random.default_rng(seed)
    size = len(candidates) if num_clips == -1 else min(num_clips, len(candidates))
    chosen = rng.choice(candidates, size=size, replace=False)
    return [str(c) for c in chosen]


# ----------------------------------------------------------------------------
# Per-clip chunk loading (CPU preprocessing) → list of chunk dicts for the VLM
# ----------------------------------------------------------------------------

def _load_clip_chunks(
    clip_id: str,
    dataset_root: str,
    num_chunks: int,
    chunk_seconds: float,
    native_fps_override: float,
    route_store: Dict[str, str],
    bev_store,
) -> List[Dict]:
    """Return a list of chunk dicts for each upcoming chunk of this clip.

    Frame selection mirrors NvidiaEpisodeDataset exactly:
      native_fps=10, target_video_fps=1, frame_skip=10, frames_per_chunk=4
      video_frame_indices = [i * frame_skip for i in range(num_chunks * frames_per_chunk + 1)]
      chunk ci uses video_frame_indices[ci * frames_per_chunk]  (indices 0, 4, 8, 12, 16)
    """
    video_path = clip_video_path(dataset_root, clip_id)
    if not os.path.exists(video_path):
        log.warning(f"[{clip_id}] missing video, skipped.")
        return []

    try:
        frames, _ = load_all_frames_rgb(video_path)
    except Exception:
        log.error(f"[{clip_id}] video load failed:\n{traceback.format_exc()}")
        return []

    total = len(frames)

    # Fixed sampling parameters — must match NvidiaEpisodeDataset
    native_fps = 10
    target_fps = 1
    frame_skip = native_fps // target_fps
    frames_per_chunk = int(chunk_seconds * target_fps)
    chunk_size_native = int(chunk_seconds * native_fps)

    # 1-Hz frame index array (same logic as training dataset with start=0)
    num_1hz_frames = num_chunks * frames_per_chunk + 1       # 21
    video_frame_indices = [min(i * frame_skip, total - 1) for i in range(num_1hz_frames)]

    out: List[Dict] = []
    for ci in range(num_chunks):
        key = _chunk_key(clip_id, ci)
        route = route_store.get(key)
        if route is None:
            continue

        if ci * chunk_size_native + chunk_size_native > total:
            continue  # clip too short for this chunk

        # Frame the model sees at the start of chunk ci in training:
        # index ci*frames_per_chunk in the 1-Hz sequence (0, 4, 8, 12, 16)
        seq_idx = ci * frames_per_chunk
        current_frame = frames[video_frame_indices[seq_idx]]

        is_start = (ci == 0)
        hist_bev = None
        if not is_start:
            prev_key = _chunk_key(clip_id, ci - 1)
            if bev_store is not None and prev_key in bev_store:
                hist_bev = PILImage.fromarray(bev_store[prev_key])
            elif bev_store is not None:
                log.warning(f"[{clip_id} c{ci}] BEV of previous chunk {ci-1} missing.")

        out.append({
            "chunk_key":         key,
            "clip_id":           clip_id,
            "chunk_idx":         ci,
            "route":             route,
            "is_start_of_clip":  is_start,
            "current_frame":     current_frame,
            "history_bev_image": hist_bev,
        })
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
        default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--dataset_root",
        default="./data/physicalai/front")
    parser.add_argument("--clip_index",
        default="./data/physicalai/clip_index.parquet")
    parser.add_argument("--output_dir",
        default=None)
    parser.add_argument("--num_clips", type=int, default=-1,
        help="Number of clips to process. -1 = all.")
    parser.add_argument("--num_chunks", type=int, default=5)
    parser.add_argument("--chunk_seconds", type=float, default=4.0)
    parser.add_argument("--native_fps_override", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tensor_parallel_size", type=int, default=8,
        help="Number of GPUs for vLLM tensor parallelism.")
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=64,
        help="Chunks per vLLM generate() call.")
    parser.add_argument("--max_images_per_prompt", type=int, default=3,
        help="Upper bound for prompt images (1 BEV + 1 frame = 2 by default).")
    parser.add_argument("--clip_ids_file", default=None,
        help="Path to a JSON file containing a list of clip_ids.")
    parser.add_argument("--bev_store",
        default="bev_store.npz",
        help="Path to bev_store.npz (from generate_bev.py).")
    parser.add_argument("--route_store",
        default="route_store.json",
        help="Path to route_store JSON (from generate_route.py).")
    parser.add_argument("--output_name",
        default="scene_evolving_guidance.json",
        help="Filename (under output_dir) for the output {clip_id: {chunk_idx: caption}} JSON.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Resolve clip IDs ---
    if args.clip_ids_file and os.path.exists(args.clip_ids_file):
        with open(args.clip_ids_file) as f:
            clip_ids = list(json.load(f))
        if args.num_clips != -1:
            clip_ids = clip_ids[: args.num_clips]
        log.info(f"Clip IDs loaded from {args.clip_ids_file} ({len(clip_ids)} clips).")
    else:
        clip_ids = select_demo_clips(
            args.clip_index, args.dataset_root, args.num_clips, seed=args.seed
        )
    log.info(f"{len(clip_ids)} clips selected.")

    # --- Load route store (required) ---
    if not os.path.exists(args.route_store):
        raise FileNotFoundError(
            f"Route store not found: {args.route_store}. "
            f"Run generate_route.py first."
        )
    with open(args.route_store) as f:
        route_store: Dict[str, str] = json.load(f)
    log.info(f"Route store: {args.route_store} ({len(route_store)} labels)")

    # --- Load BEV store (required for chunks 1..N-1) ---
    bev_store = None
    if args.bev_store and os.path.exists(args.bev_store):
        bev_store = np.load(args.bev_store, allow_pickle=False)
        log.info(f"BEV store: {args.bev_store} ({len(bev_store.files)} entries)")
    else:
        log.warning(f"BEV store not found at {args.bev_store}, running without BEV history.")

    # --- Load vLLM model ---
    captioner = VLLMNavCaptioner(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_images_per_prompt=args.max_images_per_prompt,
    )

    # --- Stream chunks through the VLM ---
    all_captions: Dict[str, Dict[int, str]] = {}
    pending: List[Dict] = []

    def _flush(batch: List[Dict]):
        if not batch:
            return
        try:
            result = captioner.caption_batch(batch)
        except Exception:
            log.error(f"Batch of {len(batch)} chunks failed:\n{traceback.format_exc()}")
            return
        for ch in batch:
            cap = result.get(ch["chunk_key"])
            if cap is None:
                continue
            all_captions.setdefault(ch["clip_id"], {})[ch["chunk_idx"]] = cap

    with tqdm(total=len(clip_ids), desc="clips", unit="clip") as pbar:
        for clip_id in clip_ids:
            chunks = _load_clip_chunks(
                clip_id, args.dataset_root,
                args.num_chunks, args.chunk_seconds,
                args.native_fps_override,
                route_store, bev_store,
            )
            pending.extend(chunks)
            pbar.update(1)

            while len(pending) >= args.batch_size:
                batch = pending[: args.batch_size]
                pending = pending[args.batch_size:]
                _flush(batch)
                pbar.set_postfix({"clips_done": len(all_captions), "queued": len(pending)})
        _flush(pending)
        pending = []

    serializable: Dict[str, Dict[str, str]] = {
        cid: {str(ci): cap for ci, cap in sorted(chunk_map.items())}
        for cid, chunk_map in all_captions.items()
    }
    out_json = out_dir / args.output_name
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    total_chunks = sum(len(v) for v in all_captions.values())
    log.info(f"Saved {len(all_captions)} clips / {total_chunks} chunks → {out_json}")


if __name__ == "__main__":
    main()
