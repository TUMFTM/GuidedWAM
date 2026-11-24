"""
Video loading and per-chunk frame extraction for the NVIDIA AV front-camera
dataset.

Each clip is a ~20 s MP4 at 10 Hz. We split it into ``num_chunks`` non-overlapping
windows of ``chunk_seconds`` seconds (default 5 × 4 s) and, inside each window,
pick ``frames_per_chunk`` frames uniformly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class VideoChunk:
    chunk_idx: int              # 0..num_chunks-1
    start_frame: int            # inclusive, in native fps
    end_frame: int              # exclusive, in native fps
    sample_indices: List[int]   # indices of frames actually fed to the VLM
    frames: List[np.ndarray]    # sampled frames, HxWx3 uint8 RGB


def clip_video_path(dataset_root: str, clip_id: str) -> str:
    return os.path.join(dataset_root, clip_id, "camera_front_wide_120fov.mp4")


def load_all_frames_rgb(video_path: str) -> Tuple[List[np.ndarray], float]:
    """Decode *video_path* into RGB frames. Returns (frames, native_fps)."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, float(fps)


def extract_chunks(
    frames: List[np.ndarray],
    native_fps: float = 10.0,
    chunk_seconds: float = 4.0,
    num_chunks: int = 5,
    frames_per_chunk: int = 8,
) -> List[VideoChunk]:
    """Split frames into chunks and sample ``frames_per_chunk`` per chunk."""
    if frames_per_chunk < 1:
        raise ValueError("frames_per_chunk must be >= 1")

    chunk_size = int(round(chunk_seconds * native_fps))
    if chunk_size < frames_per_chunk:
        raise ValueError(
            f"chunk_size ({chunk_size}) < frames_per_chunk ({frames_per_chunk})"
        )

    sample_offsets = np.linspace(
        0, chunk_size - 1, frames_per_chunk, dtype=int
    ).tolist()

    chunks: List[VideoChunk] = []
    total = len(frames)
    for c in range(num_chunks):
        start = c * chunk_size
        end = start + chunk_size
        if end > total:
            break
        idxs = [start + off for off in sample_offsets]
        chunk_frames = [frames[i] for i in idxs]
        chunks.append(
            VideoChunk(
                chunk_idx=c,
                start_frame=start,
                end_frame=end,
                sample_indices=idxs,
                frames=chunk_frames,
            )
        )
    return chunks
