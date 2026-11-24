"""Instability search: clean-estimate traces on DriveWAM failure scenes under uniform (constant) vs
scheduled (trapezoid) guidance across a strength ladder. One npz per scene:
  sigma (S,), off_trace (S,8,3), off_final (8,3), const_trace_<lam>/const_final_<lam>, sched_trace_<lam>/
  sched_final_<lam>, plus the oracle corner energy of each final plan.
Usage (cwd = drivewam repo, drivewam venv): TOKENS_FILE=... OUT_DIR=... LAMS=4,8,16,32,64,128 python x0_trace_batch.py
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from guidedwam import paths
sys.path.insert(0, str(paths.DRIVEWAM))
os.environ.setdefault("DRIVEWAM_TEXT_ENCODER_DEVICE", "cpu")

import navsim
assert "third_party/navsim/" in str(navsim.__file__)
from hydra.utils import instantiate
from omegaconf import OmegaConf
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader
from src.navsim.eval import NavSimEvaluator, VA_CONFIGS
from src.navsim.guidance import GuidanceSchedule
from guidedwam.energies.navhard_fp import CornerDrivableEnergy

TOKENS = json.load(open(os.environ["TOKENS_FILE"]))
OUT = Path(os.environ["OUT_DIR"]); OUT.mkdir(parents=True, exist_ok=True)
LAMS = [float(v) for v in os.environ.get("LAMS", "4,8,16,32,64,128").split(",")]
MARGIN = float(os.environ.get("CORNER_MARGIN", "0.5"))
ROOT = paths.NAVSIM_DATA; SIMWAM = paths.SIMWAM
FIELDS = Path(os.environ.get("FIELDS_DIR", str(paths.EXP / "navhard_fields")))
# Stabilization ablation (ARMS, default the Figure 3 pair): const = uniform window, sched = trapezoid (the method),
# nobound = trapezoid without the velocity bound, norm = trapezoid with the gradient normalized, its strength
# multiplied by NORM_SCALE so that the mean correction matches the method's. GUIDANCE_STATS=1 stores per-step stats.
ARMS = os.environ.get("ARMS", "const,sched").split(",")
ARM_DEF = {"const": ("constant", {}), "sched": ("asymmetric", {}),
           "nobound": ("asymmetric", {"clip_rel": 1e9}), "norm": ("asymmetric", {"normalize": True})}
NORM_SCALE = float(os.environ.get("NORM_SCALE", "1.0"))
from guidedwam.sampler.guidance import STATS as _STATS

sf = instantiate(OmegaConf.load(
    paths.NAVSIM / "navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml"))
loader = SceneLoader(data_path=ROOT / "navsim_logs/test", original_sensor_path=ROOT / "sensor_blobs/test",
                     scene_filter=sf, synthetic_sensor_path=ROOT / "navhard_two_stage/sensor_blobs",
                     synthetic_scenes_path=ROOT / "navhard_two_stage/synthetic_scene_pickles",
                     sensor_config=SensorConfig(cam_f0=True, cam_l0=False, cam_l1=False, cam_l2=False, cam_r0=False,
                                                cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False))
config = VA_CONFIGS["navsim_cfg"]
evaluator = NavSimEvaluator(config, checkpoint_path="./weights/DriveWAM/navsim_ckpt", attn_mode="torch", device="cuda:0")
num_frames = 1 + len(list(config.future_image_indices))
chunk_steps, action_dim = int(config.action_chunk_steps), int(config.action_dim)
channel_ids, prompts, hist_dim = list(config.used_action_channel_ids), list(config.command_prompts), int(config.history_num_frames)
actions = torch.zeros(2, chunk_steps, action_dim)
mask = torch.zeros_like(actions, dtype=torch.bool); mask[1, :, channel_ids] = True
ACTIONS = actions.permute(2, 0, 1).unsqueeze(-1).contiguous().unsqueeze(0)
MASK = mask.permute(2, 0, 1).unsqueeze(-1).contiguous().unsqueeze(0)


def make_batch(token):
    scene = loader.get_scene_from_token(token)
    cur = max(0, min(int(scene.scene_metadata.num_history_frames) - 1, len(scene.frames) - 1))
    img = scene.frames[cur].cameras.cam_f0.image
    h, w = img.shape[:2]
    img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    images = torch.as_tensor(np.stack([img] * num_frames), dtype=torch.uint8).permute(0, 3, 1, 2).contiguous().unsqueeze(0)
    hist = torch.as_tensor(scene.get_history_trajectory().poses, dtype=torch.float32)[-hist_dim:]
    if hist.shape[0] < hist_dim:
        hist = torch.cat([torch.zeros(hist_dim - hist.shape[0], hist.shape[1]), hist], 0)
    ego = scene.frames[cur].ego_status
    cmd = torch.as_tensor(ego.driving_command, dtype=torch.float32); command_id = int(torch.argmax(cmd).item())
    return {"images": images, "prompt": [prompts[min(command_id, len(prompts) - 1)]],
            "command_id": torch.tensor([command_id], dtype=torch.long),
            "ego_velocity_acc": torch.cat([torch.as_tensor(ego.ego_velocity, dtype=torch.float32),
                                           torch.as_tensor(ego.ego_acceleration, dtype=torch.float32)], 0).unsqueeze(0),
            "history_poses": hist.unsqueeze(0), "actions": ACTIONS, "actions_mask": MASK, "scene_token": [token]}


def run(batch, energy, mode, lam, clip_rel=12.0, normalize=False):
    ens = None if mode == "off" else [energy]
    evaluator.x0_trace = []
    os.environ["GUIDANCE_NORMALIZE"] = "1" if normalize else "0"
    pred, _ = evaluator.predict(batch, num_video_steps=3, num_action_steps=10, seed=0, video_denoise_fraction=0.6,
                                energies=ens, guidance_schedule=GuidanceSchedule(mode=mode, lam_max=lam) if ens else None,
                                guidance_clip_rel=clip_rel)
    final = evaluator.pred_traj_to_poses(evaluator.actions_to_trajectory(pred)).astype(np.float32)
    sig = np.array([s for s, _ in evaluator.x0_trace], np.float32)
    tr = np.stack([x for _, x in evaluator.x0_trace]).astype(np.float32)
    evaluator.x0_trace = None
    return sig, tr, final


n_ok = n_fail = 0
for tok in TOKENS:
    outp = OUT / f"x0_{tok}.npz"
    if outp.exists():
        n_ok += 1; continue
    try:
        batch = make_batch(tok)
        d = np.load(FIELDS / f"ora_{tok}.npz")
        energy = CornerDrivableEnergy(d["dist_out"], weight=1.0, margin=MARGIN)
        out = {"token": tok, "margin": MARGIN, "lams": np.array(LAMS, np.float32)}
        sig, tr, fin = run(batch, energy, "off", 0.0)
        out["sigma"], out["off_trace"], out["off_final"] = sig, tr, fin
        for lam in LAMS:
            for key in ARMS:
                mode, kw = ARM_DEF[key]
                _STATS.clear()
                _, tr, fin = run(batch, energy, mode, lam * (NORM_SCALE if key == "norm" else 1.0), **kw)
                out[f"{key}_trace_{lam:g}"], out[f"{key}_final_{lam:g}"] = tr, fin
                if _STATS:   # per guided step: raw gradient norm, applied correction / velocity, bound active
                    out[f"{key}_stats_{lam:g}"] = np.array([[s["g_raw"], s["ratio"], s["bind"]] for s in _STATS],
                                                           np.float32)
        with torch.no_grad():
            for k in [k for k in out if k.endswith("final") or "_final_" in k]:
                out["E_" + k] = float(energy(torch.as_tensor(out[k])[None])[0])
        np.savez(outp, **out); n_ok += 1
        print(f"{n_ok} {tok} E_off={out['E_off_final']:.3f} " +
              " ".join(f"E_{k}4={out[f'E_{k}_final_4']:.3f}" for k in ARMS if f"E_{k}_final_4" in out), flush=True)
    except Exception as e:
        n_fail += 1; print(f"FAILED {tok}: {type(e).__name__}: {e}", flush=True)
print(f"x0 batch done ok={n_ok} fail={n_fail} -> {OUT}")
