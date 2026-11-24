"""DriveWAM trajectory prediction over NAVSIM v2 ``navhard_two_stage``.

DriveWAM's own pipeline is v1-only (process_data -> pkl -> eval), so this
adapter mirrors SimWAM's ``predict_navhard.py``: it drives the **v2**
SceneLoader (original stage-one + synthetic stage-two scenes) directly and
builds the batch that ``NavSimEvaluator.predict`` consumes, replicating
``NavSimEpisodeDataset.__getitem__``:

- ``images``: current CAM_F0 frame, 2x-downsampled (INTER_AREA) like
  process_data, replicated across the (1 + future) frame slots — predict()
  encodes only the current frame; the future slots are training-time GT.
- ``history_poses``: last 4 poses; ``ego_velocity_acc``; ``command_id``;
  ``prompt`` from config.command_prompts; ``actions``/``actions_mask``:
  zero/dummy chunk-1 tensors (predict zeroes the action condition anyway).

Inference uses the reproduce-gate settings (3 video steps, 10 action steps,
seed 0, denoise fraction 0.6, attn_mode=torch, cpu text encoder). Output:
one ``<token>.npy`` per scene, ``(8, 3)`` poses @ 0.5 s in the current
ego rear-axle frame — the format ``NpyTrajectoryAgent`` consumes.

Run from the drivewam repo root with the drivewam venv and PYTHONPATH
prepending ``third_party/navsim``.
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from guidedwam import paths
sys.path.insert(0, str(paths.DRIVEWAM))
os.environ.setdefault("DRIVEWAM_TEXT_ENCODER_DEVICE", "cpu")

import navsim  # must resolve to third_party/navsim
assert "third_party/navsim/" in str(navsim.__file__), f"navsim resolved to {navsim.__file__}"
from hydra.utils import instantiate
from omegaconf import OmegaConf
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader

from src.navsim.eval import NavSimEvaluator, VA_CONFIGS

ROOT = paths.NAVSIM_DATA
SIMWAM = paths.SIMWAM
OUT = Path(os.environ.get(
    "PRED_DIR", str(paths.EXP / "navhard_pred/drivewam/pred_actions")))
OUT.mkdir(parents=True, exist_ok=True)

scene_filter = instantiate(OmegaConf.load(
    paths.NAVSIM / "navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml"))
assert getattr(scene_filter, "include_synthetic_scenes", False)
loader = SceneLoader(
    data_path=ROOT / "navsim_logs/test",
    original_sensor_path=ROOT / "sensor_blobs/test",
    scene_filter=scene_filter,
    synthetic_sensor_path=ROOT / "navhard_two_stage/sensor_blobs",
    synthetic_scenes_path=ROOT / "navhard_two_stage/synthetic_scene_pickles",
    sensor_config=SensorConfig(cam_f0=True, cam_l0=False, cam_l1=False, cam_l2=False,
                               cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False,
                               lidar_pc=False),
)
tokens = list(loader.tokens)
if os.environ.get("TOKENS_FILE"):   # subset for search and regression checks; the paper runs use the full split
    keep = {t for t, *_ in json.load(open(os.environ["TOKENS_FILE"]))}
    tokens = [t for t in tokens if t in keep]
syn = set(loader.synthetic_scenes_tokens)
print(f"navhard tokens: {len(tokens)} (synthetic: {len(syn)})", flush=True)

config = VA_CONFIGS["navsim_cfg"]
evaluator = NavSimEvaluator(config, checkpoint_path="./weights/DriveWAM/navsim_ckpt",
                            attn_mode="torch", device="cuda:0")

num_frames = 1 + len(list(config.future_image_indices))
chunk_steps = int(config.action_chunk_steps)
action_dim = int(config.action_dim)
channel_ids = list(config.used_action_channel_ids)
prompts = list(config.command_prompts)
hist_dim = int(config.history_num_frames)

actions = torch.zeros(2, chunk_steps, action_dim)
mask = torch.zeros_like(actions, dtype=torch.bool)
mask[1, :, channel_ids] = True
ACTIONS = actions.permute(2, 0, 1).unsqueeze(-1).contiguous().unsqueeze(0)
MASK = mask.permute(2, 0, 1).unsqueeze(-1).contiguous().unsqueeze(0)

n_ok = n_fail = n_skip = 0
for token in tqdm(tokens):
    out_path = OUT / f"{token}.npy"
    if out_path.exists():
        n_skip += 1
        continue
    try:
        scene = loader.get_scene_from_token(token)
        cur = max(0, min(int(scene.scene_metadata.num_history_frames) - 1,
                         len(scene.frames) - 1))
        img = scene.frames[cur].cameras.cam_f0.image
        if img is None:
            raise ValueError("cam_f0 image is None")
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        images = torch.as_tensor(np.stack([img] * num_frames), dtype=torch.uint8)
        images = images.permute(0, 3, 1, 2).contiguous().unsqueeze(0)

        hist = torch.as_tensor(scene.get_history_trajectory().poses,
                               dtype=torch.float32)[-hist_dim:]
        if hist.shape[0] < hist_dim:
            hist = torch.cat([torch.zeros(hist_dim - hist.shape[0], hist.shape[1]), hist], 0)

        ego = scene.frames[cur].ego_status
        vel = torch.as_tensor(ego.ego_velocity, dtype=torch.float32)
        acc = torch.as_tensor(ego.ego_acceleration, dtype=torch.float32)
        cmd = torch.as_tensor(ego.driving_command, dtype=torch.float32)
        command_id = int(torch.argmax(cmd).item())

        batch = {
            "images": images,
            "prompt": [prompts[min(command_id, len(prompts) - 1)]],
            "command_id": torch.tensor([command_id], dtype=torch.long),
            "ego_velocity_acc": torch.cat([vel, acc], 0).unsqueeze(0),
            "history_poses": hist.unsqueeze(0),
            "actions": ACTIONS,
            "actions_mask": MASK,
            "scene_token": [token],
        }
        pred_actions, _ = evaluator.predict(batch, num_video_steps=3,
                                            num_action_steps=10, seed=0,
                                            video_denoise_fraction=0.6)
        poses = evaluator.pred_traj_to_poses(evaluator.actions_to_trajectory(pred_actions))
        poses = np.asarray(poses, dtype=np.float32)
        assert poses.shape == (8, 3), poses.shape
        np.save(out_path, poses)
        n_ok += 1
    except Exception as e:
        print(f"FAILED {token}: {type(e).__name__}: {e}", flush=True)
        n_fail += 1

print(f"done: ok={n_ok} fail={n_fail} skip={n_skip} -> {OUT}")
