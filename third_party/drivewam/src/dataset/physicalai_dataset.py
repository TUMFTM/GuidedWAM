import json
import logging
import os
import pickle
import time
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _load_all_video_frames(video_path, height, width):
    """Read all frames from *video_path*, resize to (*height*, *width*), and cache."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


class PhysicalAIEpisodeDataset(torch.utils.data.Dataset):
    """
    Dataset for NVIDIA PhysicalAI-Autonomous-Vehicles-Front.

    Reads raw video clips (MP4) and ego annotations (PKL) directly.
    Each ~20 s clip provides one sample; a random 12 s window is selected
    per epoch (data augmentation).

    Video frames are downsampled from 10 Hz to ``target_video_fps`` (default 1 Hz).
    Actions remain at the native 10 Hz resolution.

    Returned batch dict:
      - ``images``: (num_video_frames, 3, H, W) -- e.g. (13, 3, 256, 448)
      - ``actions``: (action_dim, 4, 40, 1) -- 1 cond + 3 future chunks, 40 steps each
    """

    def __init__(self, config):
        self.config = config

        # ----- Paths -----
        self.data_root = Path(config.dataset_path)
        clip_index_path =  Path(config.clip_index_path)

        # ----- Split / subset -----
        split = getattr(config, "split", "train")
        max_clips = int(getattr(config, "max_clips", 5000))

        clip_df = pd.read_parquet(clip_index_path)
        split_clips = clip_df.loc[
            (clip_df["split"] == split) & clip_df["clip_is_valid"]
        ]

        # Fast set-based existence check
        existing_clip_ids = set(os.listdir(str(self.data_root)))
        clip_ids_csv = getattr(config, "clip_ids_csv", None)
        if clip_ids_csv:
            csv_df = pd.read_csv(clip_ids_csv)
            csv_ids = [str(cid) for cid in csv_df["clip_id"].tolist()]
            available_clips = [cid for cid in csv_ids if cid in existing_clip_ids][:max_clips]
        else:
            available_clips = [
                cid for cid in split_clips.index if cid in existing_clip_ids
            ][:max_clips]

        if len(available_clips) == 0:
            raise FileNotFoundError(
                f"No PhysicalAI clips found for split={split} under {self.data_root}"
            )

        self.clip_ids = available_clips

        # ----- Timing parameters -----
        self.video_fps = 10  # native annotation / video frame rate
        self.target_video_fps = int(getattr(config, "target_video_fps", 1))
        self.sample_duration_s = int(getattr(config, "sample_duration_s", 12))

        self.frame_skip = self.video_fps // self.target_video_fps
        self.num_video_frames = self.sample_duration_s * self.target_video_fps + 1
        self.num_action_steps = self.sample_duration_s * self.video_fps

        # ----- Model / action parameters -----
        self.action_chunk_count = int(config.action_chunk_count)
        self.action_channel_ids = list(config.used_action_channel_ids)
        self.action_norm_method = str(config.action_norm_method)
        norm_stat = config.norm_stat
        self.actions_q01 = torch.tensor(norm_stat["q01"], dtype=torch.float32)
        self.actions_q99 = torch.tensor(norm_stat["q99"], dtype=torch.float32)
        self.prompt = str(config.prompt)
        self.image_height = int(config.height)
        self.image_width = int(config.width)

        # ----- Per-chunk prompts (VLM-generated) -----
        prompts_dir = getattr(config, "prompts_dir", None)
        self.prompts_dir: Optional[Path] = None
        self.prompts_store: Optional[Dict[str, Dict[str, str]]] = None
        if prompts_dir:
            self.prompts_dir = Path(prompts_dir)
            if not self.prompts_dir.is_dir():
                raise FileNotFoundError(f"prompts_dir not found: {prompts_dir}")
            logger.info(f"Per-clip prompts dir: {prompts_dir}")
        else:
            prompts_store_path = getattr(config, "prompts_store_path", None)
            if prompts_store_path:
                if not os.path.exists(prompts_store_path):
                    raise FileNotFoundError(
                        f"prompts_store_path not found: {prompts_store_path}"
                    )
                with open(prompts_store_path, "r") as f:
                    self.prompts_store = json.load(f)
                logger.info(
                    f"Loaded per-chunk prompts store with {len(self.prompts_store)} clips "
                    f"from {prompts_store_path}"
                )

        # ----- Constrained window sampling -----
        self.total_chunks_in_clip = int(getattr(config, "total_chunks_in_clip", 5))
        self.num_future_chunks = self.action_chunk_count - 1
        if self.num_future_chunks > self.total_chunks_in_clip:
            raise ValueError(
                f"num_future_chunks={self.num_future_chunks} exceeds "
                f"total_chunks_in_clip={self.total_chunks_in_clip}"
            )
        self.max_abs_chunk_start = self.total_chunks_in_clip - self.num_future_chunks

        logger.info(
            f"PhysicalAIEpisodeDataset: {len(self.clip_ids)} clips (= samples), "
            f"{self.num_video_frames} video frames @ {self.target_video_fps} Hz, "
            f"{self.num_action_steps} action steps @ {self.video_fps} Hz, "
            f"{self.num_future_chunks} future chunks, "
            f"abs_chunk_start ∈ [0, {self.max_abs_chunk_start}]"
        )

    # ------------------------------------------------------------------
    # Length
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.clip_ids)

    # ------------------------------------------------------------------
    # Static / pure helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _se3_inv(T):
        """Analytic SE(3) inverse: [R|t]^-1 = [R^T | -R^T t]."""
        R, t = T[:3, :3], T[:3, 3]
        T_inv = np.eye(4, dtype=T.dtype)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -(R.T @ t)
        return T_inv


    # ------------------------------------------------------------------
    # Action tensor construction
    # ------------------------------------------------------------------

    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        q01 = self.actions_q01.to(actions.device)
        q99 = self.actions_q99.to(actions.device)
        n = len(q01)
        actions[..., :n] = (actions[..., :n] - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        return actions

    def _build_action_tensor(self, future_trajectory):
        """Build action tensor from future trajectory.

        Actions are split into ``action_chunk_count`` chunks (1 cond + N future)
        of ``action_chunk_steps`` steps each, aligned with latent frames.

        Returns
        -------
        actions : (action_dim, chunk_count, action_chunk_steps, 1)
        actions_mask : same shape, bool
        """
        action_chunk_steps = self.num_action_steps // (self.action_chunk_count - 1)
        actions = torch.zeros(
            self.action_chunk_count,
            action_chunk_steps,
            self.config.action_dim,
            dtype=torch.float32,
        )
        actions_mask = torch.zeros_like(actions, dtype=torch.bool)

        action_channel_ids = self.action_channel_ids[:3]
        if len(action_channel_ids) != 3:
            raise ValueError("Expected exactly 3 action channels for dx/dy/dheading")

        prev_pose = torch.zeros(3, dtype=torch.float32)
        deltas = []
        for step in range(self.num_action_steps):
            pose = torch.as_tensor(future_trajectory[step]["pose"], dtype=torch.float32)
            deltas.append(pose - prev_pose)
            prev_pose = pose

        deltas = torch.stack(deltas, dim=0)  # (num_action_steps, 3)
        deltas = self._normalize_actions(deltas)

        # Split into future chunks (chunks 1..N, chunk 0 stays zeros as condition)
        num_future_chunks = self.action_chunk_count - 1
        for chunk_idx in range(num_future_chunks):
            start = chunk_idx * action_chunk_steps
            end = start + action_chunk_steps
            actions[chunk_idx + 1, :, action_channel_ids] = deltas[start:end]
            actions_mask[chunk_idx + 1, :, action_channel_ids] = True

        # -> (action_dim, chunk_count, action_chunk_steps, 1)
        actions = actions.permute(2, 0, 1).unsqueeze(-1).contiguous()
        actions_mask = actions_mask.permute(2, 0, 1).unsqueeze(-1).contiguous()
        return actions, actions_mask

    # ------------------------------------------------------------------
    # Data I/O helpers
    # ------------------------------------------------------------------

    def _load_ego_data(self, clip_id, max_retries=3, retry_delay=1.0):
        ego_pkl_path = self.data_root / clip_id / "camera_front_wide_120fov_ego.pkl"
        for attempt in range(max_retries):
            try:
                with open(ego_pkl_path, "rb") as f:
                    return pickle.load(f)
            except (OSError, IOError) as e:
                logger.warning(
                    f"[Retry {attempt + 1}/{max_retries}] Failed to load ego data: "
                    f"{ego_pkl_path}\n  Error: {type(e).__name__}: {e}\n  PID: {os.getpid()}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
        raise OSError(
            f"Failed to load ego data after {max_retries} retries: {ego_pkl_path}"
        )

    def _load_video_frames(self, clip_id, frame_indices):
        """Load specific video frames (by annotation index) with caching."""
        video_path = str(self.data_root / clip_id / "camera_front_wide_120fov.mp4")
        all_frames = _load_all_video_frames(
            video_path, self.image_height, self.image_width
        )

        frames = []
        for fi in frame_indices:
            if 0 <= fi < len(all_frames):
                frames.append(all_frames[fi])
            elif frames:
                frames.append(frames[-1])  # repeat last valid frame
            else:
                frames.append(
                    np.zeros(
                        (self.image_height, self.image_width, 3), dtype=np.uint8
                    )
                )
        return np.stack(frames, axis=0)  # (T, H, W, 3)

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        clip_id = self.clip_ids[idx]

        try:
            ego_data = self._load_ego_data(clip_id)
        except Exception as e:
            logger.error(
                f"[PhysicalAIEpisodeDataset] Unrecoverable error loading ego data.\n"
                f"  Index: {idx}, clip_id: {clip_id}\n"
                f"  Error: {type(e).__name__}: {e}\n"
                f"  Traceback:\n{traceback.format_exc()}"
            )
            raise

        try:
            N = len(ego_data["timestamps_us"])
            poses = ego_data["ego_pose__world_from_rig"]  # (N, 4, 4)
            velocities = ego_data["ego_velocity_rig"]  # (N, 3)
            accelerations = ego_data["ego_acceleration_rig"]  # (N, 3)
            curvature = ego_data["ego_curvature"]  # (N,)

            # ---- Constrained window sampling ----
            # Only allow starts aligned to absolute chunk boundaries so each
            # future chunk maps 1-to-1 to one of the pre-computed per-chunk
            # VLM prompts.  E.g. for a 12 s window in a 20 s clip:
            abs_chunk_start = int(
                np.random.randint(0, self.max_abs_chunk_start + 1)
            )
            start = abs_chunk_start * int(self.config.action_chunk_steps)

            # Clip too short: fall back to the earliest valid start.
            if start + self.num_action_steps + 1 > N:
                abs_chunk_start = 0
                start = 0
                if self.num_action_steps + 1 > N:
                    raise ValueError(
                        f"Clip {clip_id} too short: N={N}, "
                        f"need at least {self.num_action_steps + 1} frames"
                    )

            # ---- Video frames at target FPS (e.g. 2 Hz) ----
            video_frame_indices = [
                min(start + i * self.frame_skip, N - 1)
                for i in range(self.num_video_frames)
            ]
            frames = self._load_video_frames(clip_id, video_frame_indices)

            # ---- Coordinate transform: world SE(3) -> ego-local ----
            T_curr_inv = self._se3_inv(poses[start])
            R_curr = poses[start, :3, :3]  # rotation: world <- body at start frame

            # Future trajectory (num_action_steps steps at 10 Hz)
            future_trajectory = []
            for k in range(start + 1, start + self.num_action_steps + 1):
                k_safe = min(k, N - 1)
                T_rel = T_curr_inv @ poses[k_safe]
                R_k = poses[k_safe, :3, :3]  # rotation: world <- body at step k
                future_trajectory.append(
                    {
                        "pose": np.array(
                            [
                                T_rel[0, 3],
                                T_rel[1, 3],
                                np.arctan2(T_rel[1, 0], T_rel[0, 0]),
                            ],
                            dtype=np.float32,
                        ),
                        "velocity": (R_k.T @ velocities[k_safe])[:2].astype(np.float32),
                        "acceleration": (R_k.T @ accelerations[k_safe])[:2].astype(np.float32),
                    }
                )

            # ---- Per-chunk ego status: one representative frame per action chunk ----
            frames_per_chunk = self.config.action_chunk_steps // self.frame_skip
            num_ego_chunks = self.action_chunk_count - 1
            ego_positions = [i * frames_per_chunk for i in range(num_ego_chunks)]

            ego_velocities = []
            ego_accelerations = []
            ego_curvatures = []
            for pos in ego_positions:
                fi = video_frame_indices[pos]
                fi_safe = min(fi, N - 1)
                R_fi = poses[fi_safe, :3, :3]
                vel = (R_fi.T @ velocities[fi_safe])[:2].astype(np.float32)
                acc = (R_fi.T @ accelerations[fi_safe])[:2].astype(np.float32)
                kap = float(curvature[fi_safe])
                ego_velocities.append(vel)
                ego_accelerations.append(acc)
                ego_curvatures.append(kap)

            ego_velocity_acc = torch.cat([
                torch.as_tensor(np.stack(ego_velocities), dtype=torch.float32),
                torch.as_tensor(np.stack(ego_accelerations), dtype=torch.float32),
            ], dim=-1)  # (num_ego_chunks, 4)
            ego_curvature = torch.as_tensor(
                np.array(ego_curvatures), dtype=torch.float32
            )  # (num_ego_chunks,)

            # ---- Assemble tensors ----
            images = torch.as_tensor(frames, dtype=torch.uint8)
            images = images.permute(0, 3, 1, 2).contiguous()  # (T, 3, H, W)

            actions, actions_mask = self._build_action_tensor(future_trajectory)

            # ---- Per-chunk prompts ----
            prompts: List[str] = [""]
            if self.prompts_dir is not None:
                prompt_file = self.prompts_dir / f"{clip_id}.json"
                if prompt_file.exists():
                    with open(prompt_file, "r") as _f:
                        clip_store = json.load(_f).get("chunks", {})
                else:
                    clip_store = {}
            elif self.prompts_store is not None:
                clip_store = self.prompts_store.get(str(clip_id), {})
            else:
                clip_store = {}
            for i in range(self.num_future_chunks):
                abs_idx = abs_chunk_start + i
                prompts.append(clip_store.get(str(abs_idx), self.prompt))

        except Exception as e:
            logger.error(
                f"[PhysicalAIEpisodeDataset] Error processing sample.\n"
                f"  Index: {idx}, clip_id: {clip_id}, start: {start}\n"
                f"  Error: {type(e).__name__}: {e}\n"
                f"  Traceback:\n{traceback.format_exc()}"
            )
            raise

        return {
            "images": images,
            "prompts": prompts,
            "curvature": ego_curvature,
            "ego_velocity_acc": ego_velocity_acc,
            "actions": actions,
            "actions_mask": actions_mask,
            "scene_token": clip_id,
            "abs_chunk_start": abs_chunk_start,
        }


class PhysicalAITestDataset(torch.utils.data.Dataset):
    """
    Test dataset for NVIDIA PhysicalAI-Autonomous-Vehicles-Front.

    Loads full ~20s clips without random windowing (unlike the training dataset
    which randomly samples 12s windows).  Each clip starts at frame 0.

    Video frames are downsampled from 10 Hz to ``target_video_fps`` (default 1 Hz,
    matching the training config).  Actions remain at native 10 Hz resolution.
    """

    def __init__(self, config, max_clips=1000, clip_ids=None, clip_prompts_path=None):
        self.config = config
        self.data_root = Path(config.dataset_path)

        if clip_ids is not None:
            existing_clip_ids = set(os.listdir(str(self.data_root)))
            available_clips = [cid for cid in clip_ids if cid in existing_clip_ids][:max_clips]
            if len(available_clips) == 0:
                raise FileNotFoundError(
                    f"None of the provided clip IDs found under {self.data_root}"
                )
        else:
            clip_index_path = Path(config.clip_index_path)

            split = getattr(config, "split", "test")
            clip_df = pd.read_parquet(clip_index_path)
            split_clips = clip_df.loc[
                (clip_df["split"] == split) & clip_df["clip_is_valid"]
            ]

            existing_clip_ids = set(os.listdir(str(self.data_root)))
            available_clips = [
                cid for cid in split_clips.index if cid in existing_clip_ids
            ][:max_clips]

            if len(available_clips) == 0:
                raise FileNotFoundError(
                    f"No NVIDIA clips found for split={split} under {self.data_root}"
                )

        self.clip_ids = available_clips

        # --- Timing parameters ---
        self.video_fps = 10  # native annotation rate
        self.target_video_fps = int(getattr(config, "target_video_fps", 1))
        self.frame_skip = self.video_fps // self.target_video_fps
        self.action_chunk_steps = int(config.action_chunk_steps)

        # --- Action normalization ---
        self.action_channel_ids = list(config.used_action_channel_ids)
        norm_stat = config.norm_stat
        self.actions_q01 = torch.tensor(norm_stat["q01"], dtype=torch.float32)
        self.actions_q99 = torch.tensor(norm_stat["q99"], dtype=torch.float32)
        self.prompt = str(config.prompt)
        self.image_height = int(config.height)
        self.image_width = int(config.width)

        # Per-chunk prompts: CLI clip_prompts_path (test set) takes priority over
        # config prompts_dir (training set).
        self.prompts_dir = None
        self.prompts_store = None
        if clip_prompts_path and os.path.exists(clip_prompts_path):
            if os.path.isdir(clip_prompts_path):
                self.prompts_dir = Path(clip_prompts_path)
            else:
                with open(clip_prompts_path, "r") as f:
                    self.prompts_store = json.load(f)
        else:
            prompts_dir = getattr(config, "prompts_dir", None)
            if prompts_dir and os.path.isdir(prompts_dir):
                self.prompts_dir = Path(prompts_dir)
            else:
                prompts_store_path = getattr(config, "prompts_store_path", None)
                if prompts_store_path and os.path.exists(prompts_store_path):
                    with open(prompts_store_path, "r") as f:
                        self.prompts_store = json.load(f)

        logger.info(
            f"PhysicalAITestDataset: {len(self.clip_ids)} clips, "
            f"target_video_fps={self.target_video_fps}, "
            f"action_chunk_steps={self.action_chunk_steps}"
        )

    def __len__(self):
        return len(self.clip_ids)

    @staticmethod
    def _se3_inv(T):
        """Analytic SE(3) inverse."""
        R, t = T[:3, :3], T[:3, 3]
        T_inv = np.eye(4, dtype=T.dtype)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -(R.T @ t)
        return T_inv

    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        q01 = self.actions_q01.to(actions.device)
        q99 = self.actions_q99.to(actions.device)
        n = len(q01)
        actions[..., :n] = (actions[..., :n] - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        return actions

    def _load_ego_data(self, clip_id, max_retries=3, retry_delay=1.0):
        ego_pkl_path = self.data_root / clip_id / "camera_front_wide_120fov_ego.pkl"
        for attempt in range(max_retries):
            try:
                with open(ego_pkl_path, "rb") as f:
                    return pickle.load(f)
            except (OSError, IOError) as e:
                logger.warning(
                    f"[Retry {attempt + 1}/{max_retries}] Failed to load ego data: "
                    f"{ego_pkl_path}\n  Error: {type(e).__name__}: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
        raise OSError(
            f"Failed to load ego data after {max_retries} retries: {ego_pkl_path}"
        )

    def _load_video_frames(self, clip_id, frame_indices):
        video_path = str(self.data_root / clip_id / "camera_front_wide_120fov.mp4")
        all_frames = _load_all_video_frames(
            video_path, self.image_height, self.image_width
        )
        frames = []
        for fi in frame_indices:
            if 0 <= fi < len(all_frames):
                frames.append(all_frames[fi])
            elif frames:
                frames.append(frames[-1])
            else:
                frames.append(
                    np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
                )
        return np.stack(frames, axis=0)

    def __getitem__(self, idx):
        clip_id = self.clip_ids[idx]

        try:
            ego_data = self._load_ego_data(clip_id)
        except Exception as e:
            logger.error(
                f"[PhysicalAITestDataset] Unrecoverable error loading ego data.\n"
                f"  Index: {idx}, clip_id: {clip_id}\n"
                f"  Error: {type(e).__name__}: {e}\n"
                f"  Traceback:\n{traceback.format_exc()}"
            )
            raise

        try:
            N = len(ego_data["timestamps_us"])
            poses = ego_data["ego_pose__world_from_rig"]  # (N, 4, 4)
            velocities = ego_data["ego_velocity_rig"]  # (N, 3)
            accelerations = ego_data["ego_acceleration_rig"]  # (N, 3)
            curvature = ego_data["ego_curvature"]  # (N,)

            start = 0  # Always start from beginning (no random window)

            # Compute how many full action chunks fit
            max_action_steps = N - 1 - start
            num_future_chunks = max_action_steps // self.action_chunk_steps
            if num_future_chunks < 1:
                raise ValueError(f"Clip {clip_id} too short: N={N}")
            num_action_steps = num_future_chunks * self.action_chunk_steps

            # Video frames at target FPS
            # Ensure (num_video_frames - 1) % 4 == 0 for clean VAE temporal compression
            frames_per_chunk = self.action_chunk_steps // self.frame_skip
            num_video_frames = num_future_chunks * frames_per_chunk + 1
            remainder = (num_video_frames - 1) % 4
            if remainder != 0:
                num_video_frames -= remainder
                num_future_chunks = (num_video_frames - 1) // frames_per_chunk
                num_action_steps = num_future_chunks * self.action_chunk_steps

            video_frame_indices = [
                min(start + i * self.frame_skip, N - 1)
                for i in range(num_video_frames)
            ]
            frames = self._load_video_frames(clip_id, video_frame_indices)

            # Coordinate transform: world -> ego-local at start frame
            T_curr_inv = self._se3_inv(poses[start])

            # Future trajectory
            future_trajectory = []
            prev_pose = np.zeros(3, dtype=np.float32)
            for k in range(start + 1, start + num_action_steps + 1):
                k_safe = min(k, N - 1)
                T_rel = T_curr_inv @ poses[k_safe]
                R_k = poses[k_safe, :3, :3]
                future_trajectory.append(
                    {
                        "pose": np.array(
                            [
                                T_rel[0, 3],
                                T_rel[1, 3],
                                np.arctan2(T_rel[1, 0], T_rel[0, 0]),
                            ],
                            dtype=np.float32,
                        ),
                        "velocity": (R_k.T @ velocities[k_safe])[:2].astype(np.float32),
                        "acceleration": (R_k.T @ accelerations[k_safe])[:2].astype(np.float32),
                    }
                )

            # Build action tensor
            action_chunk_count = num_future_chunks + 1  # 1 cond + N future
            actions = torch.zeros(
                action_chunk_count,
                self.action_chunk_steps,
                self.config.action_dim,
                dtype=torch.float32,
            )
            actions_mask = torch.zeros_like(actions, dtype=torch.bool)
            action_channel_ids = self.action_channel_ids[:3]

            prev_pose_t = torch.zeros(3, dtype=torch.float32)
            deltas = []
            for step in range(num_action_steps):
                pose = torch.as_tensor(future_trajectory[step]["pose"], dtype=torch.float32)
                deltas.append(pose - prev_pose_t)
                prev_pose_t = pose
            deltas = torch.stack(deltas, dim=0)  # (num_action_steps, 3)
            deltas = self._normalize_actions(deltas)

            for chunk_idx in range(num_future_chunks):
                s = chunk_idx * self.action_chunk_steps
                e = s + self.action_chunk_steps
                actions[chunk_idx + 1, :, action_channel_ids] = deltas[s:e]
                actions_mask[chunk_idx + 1, :, action_channel_ids] = True

            actions = actions.permute(2, 0, 1).unsqueeze(-1).contiguous()
            actions_mask = actions_mask.permute(2, 0, 1).unsqueeze(-1).contiguous()

            # Per-chunk ego status
            num_ego_chunks = num_future_chunks
            ego_positions = [i * frames_per_chunk for i in range(num_ego_chunks)]

            ego_velocities = []
            ego_accelerations = []
            ego_curvatures = []
            for pos in ego_positions:
                fi = video_frame_indices[pos]
                fi_safe = min(fi, N - 1)
                R_fi = poses[fi_safe, :3, :3]
                vel = (R_fi.T @ velocities[fi_safe])[:2].astype(np.float32)
                acc = (R_fi.T @ accelerations[fi_safe])[:2].astype(np.float32)
                kap = float(curvature[fi_safe])
                ego_velocities.append(vel)
                ego_accelerations.append(acc)
                ego_curvatures.append(kap)

            ego_velocity_acc = torch.cat([
                torch.as_tensor(np.stack(ego_velocities), dtype=torch.float32),
                torch.as_tensor(np.stack(ego_accelerations), dtype=torch.float32),
            ], dim=-1)  # (num_ego_chunks, 4)
            ego_curvature = torch.as_tensor(
                np.array(ego_curvatures), dtype=torch.float32
            )  # (num_ego_chunks,)

            # Assemble image tensor
            images = torch.as_tensor(frames, dtype=torch.uint8)
            images = images.permute(0, 3, 1, 2).contiguous()  # (T, 3, H, W)

        except Exception as e:
            logger.error(
                f"[PhysicalAITestDataset] Error processing sample.\n"
                f"  Index: {idx}, clip_id: {clip_id}\n"
                f"  Error: {type(e).__name__}: {e}\n"
                f"  Traceback:\n{traceback.format_exc()}"
            )
            raise

        # Build per-chunk prompts (test always starts at abs_chunk_start=0)
        if self.prompts_dir is not None:
            prompt_file = self.prompts_dir / f"{clip_id}.json"
            clip_store = {}
            if prompt_file.exists():
                with open(prompt_file, "r") as _f:
                    clip_store = json.load(_f).get("chunks", {})
        elif self.prompts_store is not None:
            clip_store = self.prompts_store.get(str(clip_id), {})
        else:
            clip_store = {}
        prompts = [""]  # chunk 0 = cond (empty string)
        for i in range(num_future_chunks):
            prompts.append(clip_store.get(str(i), self.prompt))

        return {
            "images": images,
            "prompts": prompts,
            "curvature": ego_curvature,
            "ego_velocity_acc": ego_velocity_acc,
            "actions": actions,
            "actions_mask": actions_mask,
            "scene_token": clip_id,
        }
