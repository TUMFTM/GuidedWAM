"""
Qwen3-VL captioner for per-chunk navigation guidance.

For each upcoming 4-second chunk the VLM is given:
  * A high-level route command {straight | left | right} — comes from GT ego
    yaw change (see generate_route.py) but is a low-bandwidth nav instruction,
    analogous to what a real routing system provides.
  * Past observations only:
      - chunk_i, i >= 1 : first frame of chunk_i + BEV of chunk_{i-1}
      - chunk_0         : the first front-cam frame of the clip; no BEV

The VLM must NOT see any frame, BEV, trajectory, or speed information that
belongs to the upcoming window itself. Its output describes what the ego
*should* do next — reasoned from the current road conditions and recent
trajectory, anchored by the route command.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

try:
    from transformers import AutoProcessor
except ImportError as e:
    raise ImportError("transformers is required.") from e

try:
    from qwen_vl_utils import process_vision_info
except ImportError as e:
    raise ImportError(
        "qwen_vl_utils is required. Install via:\n"
        "  pip install qwen-vl-utils"
    ) from e

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a navigation assistant for an autonomous-driving dataset. You "
    "generate SHORT navigation guidance for an UPCOMING 4-second driving "
    "window. You DO NOT see the upcoming window — you see only the current "
    "road conditions (one front-camera frame from just before the upcoming "
    "window), the recent ego trajectory (BEV of the previous 4 seconds when "
    "available), and a high-level route command.\n\n"
    "You receive:\n"
    "- A route command for the upcoming window: {straight | left | right}. "
    "Treat it as an authoritative navigation instruction.\n"
    "- One front-camera frame taken at the start of the upcoming 4-second window "
    "(i.e., the first observable frame of the window to be predicted), showing "
    "current road conditions. For the very first chunk, this is the initial "
    "frame of the clip.\n"
    "- A BEV trajectory map of the PREVIOUS 4-second window when available "
    "(omitted for the first chunk of the clip), showing recent ego motion and "
    "speed.\n\n"
    "Output EXACTLY 2 sentences, under 50 words total. No labels, bullets, "
    "markdown, or extra paragraphs. Present tense only.\n\n"
    "Sentence 1 — Current road context visible in the provided frame: road "
    "type (urban / suburban / highway / intersection / roundabout / parking "
    "lot), traffic elements (leading vehicles, cross-traffic, pedestrians, "
    "cyclists), traffic light state if visible, weather and lighting.\n"
    "Sentence 2 — Ego navigation guidance for the UPCOMING window: jointly "
    "reason from the road context in Sentence 1, the BEV trajectory, and the "
    "route command to determine what the ego should do next. The direction "
    "must be consistent with the route command. The speed and caution level "
    "must reflect the traffic conditions in Sentence 1 — slow and cautious "
    "near intersections, dense traffic, red lights, or pedestrians; steady on "
    "a clear road; brisk on an open highway. Capture any required interaction "
    "such as yielding, waiting, or merging. Qualitative language only; no "
    "numbers, units, or distances. Do not mention image, frame, map, or "
    "chunk indices."
)

USER_SUFFIX = (
    "Generate the 2-sentence navigation guidance for the UPCOMING 4 seconds."
)

_ROUTE_LINE = "Route command for the upcoming 4 seconds: {route}."

_BEV_CONTEXT = (
    "BEV trajectory of the PREVIOUS 4 seconds: origin at ego pose at the "
    "start of that window; green square=start, red square=end, "
    "white dots=1 s intervals, color=speed (blue=slow, yellow=fast)."
)
_CURRENT_FRAME_CONTEXT = (
    "Front-camera frame at the start of the upcoming 4-second window "
    "(first frame of the window to be predicted)."
)

_START_HEADER = (
    "This upcoming window is the START of the clip — no previous trajectory "
    "exists yet. The image below is the initial front-camera frame."
)
_START_FRAME_CONTEXT = "Initial front-camera frame at t=0 of the clip."


def _clean(text: str) -> str:
    out = text.strip()
    for prefix in ("Guidance:", "guidance:", "Caption:", "caption:",
                   "Scene:", "scene:", "Output:", "output:"):
        if out.startswith(prefix):
            out = out[len(prefix):].lstrip()
    return " ".join(out.split())


# ----------------------------------------------------------------------------
# qwen_vl_utils compatibility
# ----------------------------------------------------------------------------

_PV_SUPPORTS_KWARGS: Optional[bool] = None


def _pv_supports_kwargs() -> bool:
    global _PV_SUPPORTS_KWARGS
    if _PV_SUPPORTS_KWARGS is None:
        import inspect
        sig = inspect.signature(process_vision_info)
        _PV_SUPPORTS_KWARGS = "return_video_kwargs" in sig.parameters
    return _PV_SUPPORTS_KWARGS


def _process_vision(messages):
    out = (
        process_vision_info(messages, return_video_kwargs=True)
        if _pv_supports_kwargs()
        else process_vision_info(messages)
    )
    if isinstance(out, tuple) and len(out) == 3:
        images, videos, kwargs = out
    else:
        images, videos = out
        kwargs = {}
    kwargs = {
        k: v[0] if isinstance(v, list) and len(v) == 1 else v
        for k, v in kwargs.items()
    }
    return images, videos, kwargs


# ----------------------------------------------------------------------------
# vLLM-based captioner (high-throughput batch inference)
# ----------------------------------------------------------------------------

class VLLMNavCaptioner:
    """Leak-free Qwen3-VL captioner for navigation guidance generation.

    Each entry passed to caption_batch() is ONE upcoming chunk:
        "chunk_key"        : str     -- _chunk_key(clip_id, chunk_idx)
        "clip_id"          : str
        "chunk_idx"        : int     -- index of the UPCOMING chunk (0..N-1)
        "route"            : str     -- "straight" | "left" | "right"
        "is_start_of_clip" : bool    -- True if chunk_idx == 0
        "current_frame"    : np.ndarray  H×W×3 uint8 RGB
                             - chunk 0     : first frame of clip
                             - chunk i>=1 : first frame of chunk_i
        "history_bev_image": Optional[PIL.Image.Image]
                             - chunk 0     : None
                             - chunk i>=1 : BEV of chunk_{i-1}
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        max_new_tokens: int = 100,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        max_images_per_prompt: int = 3,
    ):
        from vllm import LLM, SamplingParams as _SP

        logger.info(f"Loading {model_path} with vLLM (tp={tensor_parallel_size}) ...")
        self._proc = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )
        self._llm = LLM(
            model=model_path,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            limit_mm_per_prompt={"image": max_images_per_prompt},
            mm_processor_kwargs={
                "min_pixels": 256 * 28 * 28,
                "max_pixels": 512 * 28 * 28,
            },
        )
        self._sp = _SP(max_tokens=max_new_tokens, temperature=0)
        logger.info("vLLM engine ready (v2 leak-free).")

    def _build_user_content(self, ch: Dict) -> list:
        content: list = []

        content.append({
            "type": "text",
            "text": _ROUTE_LINE.format(route=ch["route"]),
        })

        is_start = bool(ch.get("is_start_of_clip", False))
        frame    = ch.get("current_frame")
        bev      = ch.get("history_bev_image")
        frame_pil = Image.fromarray(frame) if frame is not None else None

        if is_start:
            content.append({"type": "text", "text": _START_HEADER})
            if frame_pil is not None:
                content.append({"type": "text",  "text":  _START_FRAME_CONTEXT})
                content.append({"type": "image", "image": frame_pil})
        else:
            if bev is not None:
                content.append({"type": "text",  "text":  _BEV_CONTEXT})
                content.append({"type": "image", "image": bev})
            if frame_pil is not None:
                content.append({"type": "text",  "text":  _CURRENT_FRAME_CONTEXT})
                content.append({"type": "image", "image": frame_pil})

        content.append({"type": "text", "text": USER_SUFFIX})
        return content

    def caption_batch(self, chunks: List[Dict]) -> Dict[str, str]:
        prompts: list = []
        ids: list = []

        for ch in chunks:
            user_content = self._build_user_content(ch)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user",   "content": user_content},
            ]

            text = self._proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            image_inputs, _video_inputs, _ = _process_vision(messages)

            mm_data: dict = {}
            if image_inputs:
                mm_data["image"] = image_inputs

            prompts.append({"prompt": text, "multi_modal_data": mm_data})
            ids.append(ch["chunk_key"])

        outputs = self._llm.generate(prompts, self._sp)
        return {cid: _clean(out.outputs[0].text) for cid, out in zip(ids, outputs)}
