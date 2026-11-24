import logging
import os
import pickle
import time
import traceback
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

class NavSimEpisodeDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        self.data_root = Path(config.dataset_path)
        self.sample_files = sorted(self.data_root.glob("sample_*.pkl"))
        if len(self.sample_files) == 0:
            raise FileNotFoundError(f"No NavSim samples found under {self.data_root}")

        self.image_future_indices = list(config.future_image_indices)
        self.action_future_steps = int(config.future_action_steps)
        self.action_chunk_count = int(config.action_chunk_count)
        self.action_chunk_steps = int(config.action_chunk_steps)
        self.command_prompts = list(config.command_prompts)
        self.action_norm_method = str(config.action_norm_method)
        norm_stat = config.norm_stat
        self.actions_q01 = torch.tensor(norm_stat["q01"], dtype=torch.float32)
        self.actions_q99 = torch.tensor(norm_stat["q99"], dtype=torch.float32)
        self.action_channel_ids = list(config.used_action_channel_ids)
        self.history_dim = int(config.history_num_frames)

        if self.action_chunk_count != 2:
            raise ValueError("NavSim action setup expects exactly 2 action chunks")
        if self.action_chunk_steps != self.action_future_steps:
            raise ValueError("action_chunk_steps must match future_action_steps for NavSim chunked actions")

    def __len__(self):
        return len(self.sample_files)

    def _driving_command_to_index(self, driving_command: torch.Tensor) -> int:
        return int(torch.argmax(driving_command).item())

    def _build_prompt(self, command_index: int) -> str:
        command_index = max(0, min(command_index, len(self.command_prompts) - 1))
        return self.command_prompts[command_index]

    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        q01 = self.actions_q01.to(actions.device)
        q99 = self.actions_q99.to(actions.device)
        n = len(q01)
        # Linear normalization to [-1, 1] using per-channel 1%/99% quantiles.
        # Consistent with how other tasks (RoboTwin, Franka) normalize actions.
        actions[..., :n] = (actions[..., :n] - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        return actions

    def _build_action_tensor(self, sample):
        future_traj = sample["future_trajectory"]
        if len(future_traj) < self.action_future_steps:
            raise ValueError(
                f"Sample only has {len(future_traj)} future steps, expected at least {self.action_future_steps}"
            )

        actions = torch.zeros(
            self.action_chunk_count,
            self.action_chunk_steps,
            self.config.action_dim,
            dtype=torch.float32,
        )
        actions_mask = torch.zeros_like(actions, dtype=torch.bool)

        action_channel_ids = self.action_channel_ids[:3]
        if len(action_channel_ids) != 3:
            raise ValueError("NavSim action setup expects exactly 3 used action channels for dx/dy/dheading")

        prev_pose = torch.zeros(3, dtype=torch.float32)
        normalized_steps = []
        for step in range(self.action_future_steps):
            future_step = future_traj[step]
            pose = torch.as_tensor(future_step["pose"], dtype=torch.float32)

            delta_pose = pose - prev_pose
            prev_pose = pose

            normalized_steps.append(delta_pose)  # [dx, dy, dheading]

        normalized_steps = torch.stack(normalized_steps, dim=0)
        normalized_steps = self._normalize_actions(normalized_steps)

        actions[1, :, action_channel_ids] = normalized_steps[: self.action_chunk_steps]
        actions_mask[1, :, action_channel_ids] = True

        actions = actions.permute(2, 0, 1).unsqueeze(-1).contiguous()
        actions_mask = actions_mask.permute(2, 0, 1).unsqueeze(-1).contiguous()
        return actions, actions_mask

    def _load_sample(self, sample_path, max_retries=3, retry_delay=1.0):
        for attempt in range(max_retries):
            try:
                with open(sample_path, "rb") as f:
                    sample = pickle.load(f)
                return sample
            except (OSError, IOError) as e:
                file_size = sample_path.stat().st_size if sample_path.exists() else "FILE_NOT_FOUND"
                logger.warning(
                    f"[Retry {attempt + 1}/{max_retries}] Failed to load sample: {sample_path}\n"
                    f"  Error: {type(e).__name__}: {e}\n"
                    f"  File exists: {sample_path.exists()}, Size: {file_size} bytes\n"
                    f"  PID: {os.getpid()}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
        raise OSError(
            f"Failed to load sample after {max_retries} retries: {sample_path} "
            f"(size={file_size})"
        )

    def __getitem__(self, idx):
        sample_path = self.sample_files[idx]
        try:
            sample = self._load_sample(sample_path)
        except Exception as e:
            logger.error(
                f"[NavSimDataset] Unrecoverable error loading sample.\n"
                f"  Index: {idx}, Total samples: {len(self.sample_files)}\n"
                f"  Path: {sample_path}\n"
                f"  File exists: {sample_path.exists()}\n"
                f"  File size: {sample_path.stat().st_size if sample_path.exists() else 'N/A'} bytes\n"
                f"  PID: {os.getpid()}\n"
                f"  Error: {type(e).__name__}: {e}\n"
                f"  Traceback:\n{traceback.format_exc()}"
            )
            raise

        try:
            images = torch.as_tensor(sample["images"], dtype=torch.uint8)
            history_poses = torch.as_tensor(sample["history_poses"], dtype=torch.float32)
            ego_status = sample["ego_status"]
            velocity = torch.as_tensor(ego_status["velocity"], dtype=torch.float32)
            acceleration = torch.as_tensor(ego_status["acceleration"], dtype=torch.float32)
            driving_command = torch.as_tensor(ego_status["driving_command"], dtype=torch.float32)
            command_id = self._driving_command_to_index(driving_command)
            sample_stem = sample_path.stem
            prompt = self._build_prompt(command_id)

            image_indices = [0] + [future_idx + 1 for future_idx in self.image_future_indices]
            selected_images = images[image_indices].permute(0, 3, 1, 2).contiguous()

            history_poses = history_poses[-self.history_dim :]
            if history_poses.shape[0] < self.history_dim:
                history_pad = torch.zeros(self.history_dim - history_poses.shape[0], history_poses.shape[1], dtype=history_poses.dtype)
                history_poses = torch.cat([history_pad, history_poses], dim=0)

            actions, actions_mask = self._build_action_tensor(sample)

            scene_token = str(sample.get("metadata", {}).get("scene_token", ""))
        except Exception as e:
            logger.error(
                f"[NavSimDataset] Error processing sample data.\n"
                f"  Index: {idx}, Path: {sample_path}\n"
                f"  Sample keys: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}\n"
                f"  Error: {type(e).__name__}: {e}\n"
                f"  Traceback:\n{traceback.format_exc()}"
            )
            raise

        return {
            "images": selected_images,
            "prompt": prompt,
            "command_id": torch.tensor(command_id, dtype=torch.long),
            "ego_velocity_acc": torch.cat([velocity, acceleration], dim=0),
            "history_poses": history_poses,
            "actions": actions,
            "actions_mask": actions_mask,
            "scene_token": scene_token,
            "sample_path": str(sample_path),
        }
