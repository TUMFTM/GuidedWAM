"""DriveWAM guided predictions on the 120 navhard joint-failure scenes.

Arms (env GUIDANCE_ARM): freespace (camera-derived DAC+NC energy, compliant)
or map (oracle lane-center energy). Guidance enters through the stock
predict() kwargs at the validated DriveWAM operating point (lam via
GuidanceSchedule(lam_max=LAM), n=10 action steps, clip_rel=12).

Saves <token>.npy (8,3) to navhard_guided/drivewam_<arm>/pred_actions.
Set DRIVEWAM_GUIDANCE_DEBUG=1 to print per-step scale diagnostics (1st scene).
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
from guidedwam.energies.kinematic import with_speed_cap
from guidedwam.energies.navhard import (FreespaceFieldEnergy, LaneCenterEnergy,
                              TrueDrivableEnergy, BoxClearanceEnergy)

ARM = os.environ.get("GUIDANCE_ARM", "freespace")
LAM = float(os.environ.get("GUIDANCE_LAM", "0.2"))
FIELDS = Path(os.environ.get("FIELDS_DIR", str(paths.EXP / "navhard_freespace120")))
OUT = Path(os.environ.get("OUT_DIR", str(paths.EXP / f"navhard_guided/drivewam_{ARM}/pred_actions")))
OUT.mkdir(parents=True, exist_ok=True)
ROOT = paths.NAVSIM_DATA
SIMWAM = paths.SIMWAM

if os.environ.get("ALL_TOKENS"):
    sample = None
else:
    sample = json.load(open(os.environ.get(
        "TOKENS_FILE", str(paths.EXP / "navhard_fail_sample120.json"))))
scene_filter = instantiate(OmegaConf.load(
    paths.NAVSIM / "navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml"))
loader = SceneLoader(
    data_path=ROOT / "navsim_logs/test", original_sensor_path=ROOT / "sensor_blobs/test",
    scene_filter=scene_filter,
    synthetic_sensor_path=ROOT / "navhard_two_stage/sensor_blobs",
    synthetic_scenes_path=ROOT / "navhard_two_stage/synthetic_scene_pickles",
    sensor_config=SensorConfig(cam_f0=True, cam_l0=False, cam_l1=False, cam_l2=False,
                               cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False),
)

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


W_DAC = float(os.environ.get("FS_W_DAC", "1.0"))
W_NC = float(os.environ.get("FS_W_NC", "1.0"))


def build_energy(token):
    """Returns a LIST of energies for the selected arm."""
    if ARM == "off":
        return None          # unguided; used to replicate the stock baseline under a new seed
    if ARM == "speedcap":
        return []            # the speed cap alone; SPEED_CAP appends it to any arm
    if ARM == "freespace":
        d = np.load(FIELDS / f"fs_{token}.npz")
        return [FreespaceFieldEnergy(d["dist_free"], d["obst_pts"], w_dac=W_DAC, w_nc=W_NC)]
    if ARM == "map":
        d = np.load(FIELDS / f"lanes_{token}.npz")
        return [LaneCenterEnergy(d["pts"])]
    if ARM == "mapdrvfp":
        from guidedwam.energies.navhard_fp import FootprintDrivableEnergy
        d = np.load(FIELDS / f"ora_{token}.npz")
        lanes = np.load(FIELDS / f"lanes_{token}.npz")["pts"]
        return [LaneCenterEnergy(lanes, weight=W_DAC),
                FootprintDrivableEnergy(d["dist_out"], weight=W_NC,
                                        ego_half_width=float(os.environ.get("EGO_HALF_W", "2.5")))]
    if ARM in ("cornerlearn", "cornerlearnoffroute"):
        # corner energy over the TRAINED camera-only signed-distance field (bevsdf_train.py).
        # Compliant: no map at inference. CORNER_MARGIN defaults high here because the learned
        # field's false positives hug the true edge (median 0.56 m inside at margin 0.5), so a wide
        # band buys recall cheaply: 42% detection at margin 0.5 -> 75% at 1.5.
        from guidedwam.energies.navhard_fp import LearnedCornerDrivableEnergy
        d = np.load(Path(os.environ.get("LEARNED_FIELDS",
                                        str(paths.EXP / "bevsdf_fields_all"))) / f"bsdf_{token}.npz")
        ens = [LearnedCornerDrivableEnergy(d["sdf"].astype(np.float32), weight=W_DAC,
                                           margin=float(os.environ.get("CORNER_MARGIN", "1.5")))]
        if ARM == "cornerlearnoffroute":
            # + DRIVING DIRECTION gate (ego centre outside on-route drivable, metres travelled).
            # Energy is ~100x the corner scale: W_NC is its RELATIVE weight (DriveLaW used 0.06).
            from guidedwam.energies.navhard_fp import OffRouteEnergy
            rd = np.load(FIELDS / f"route_{token}.npz")
            ens.append(OffRouteEnergy(rd["dist_off"], rd["in_inter"], weight=W_NC,
                                      margin=float(os.environ.get("ROUTE_MARGIN", "0.0"))))
        return ens
    if ARM in ("cornerfs", "mapcornerfs"):
        # SAME corner energy, CAMERA freespace field instead of the oracle drivable polygons
        from guidedwam.energies.navhard_fp import CornerDrivableEnergy
        d = np.load(FIELDS / f"fs_{token}.npz")
        ens = []
        if ARM == "mapcornerfs":
            ens.append(LaneCenterEnergy(np.load(FIELDS / f"lanes_{token}.npz")["pts"], weight=W_DAC))
        ens.append(CornerDrivableEnergy(d["dist_free"], weight=W_NC,
                                        margin=float(os.environ.get("CORNER_MARGIN", "0.5"))))
        return ens
    if ARM == "mapcorneroffroute":
        # all three oracle energies: lane attractor (W_DAC) + footprint barrier (CORNER_W) + direction (W_NC)
        from guidedwam.energies.navhard_fp import CornerDrivableEnergy, OffRouteEnergy
        d = np.load(FIELDS / f"ora_{token}.npz"); rd = np.load(FIELDS / f"route_{token}.npz")
        return [LaneCenterEnergy(np.load(FIELDS / f"lanes_{token}.npz")["pts"], weight=W_DAC),
                CornerDrivableEnergy(d["dist_out"], weight=float(os.environ.get("CORNER_W", "1.0")),
                                     margin=float(os.environ.get("CORNER_MARGIN", "0.5"))),
                OffRouteEnergy(rd["dist_off"], rd["in_inter"], weight=W_NC,
                               margin=float(os.environ.get("ROUTE_MARGIN", "0.0")))]
    if ARM == "mapoffroute":
        # lane attractor + DRIVING DIRECTION gate (composition table); W_NC = direction weight
        from guidedwam.energies.navhard_fp import OffRouteEnergy
        rd = np.load(FIELDS / f"route_{token}.npz")
        return [LaneCenterEnergy(np.load(FIELDS / f"lanes_{token}.npz")["pts"], weight=W_DAC),
                OffRouteEnergy(rd["dist_off"], rd["in_inter"], weight=W_NC,
                               margin=float(os.environ.get("ROUTE_MARGIN", "0.0")))]
    if ARM == "offroute":
        # DRIVING DIRECTION gate alone (metres travelled with the ego centre off the on-route drivable)
        from guidedwam.energies.navhard_fp import OffRouteEnergy
        rd = np.load(FIELDS / f"route_{token}.npz")
        return [OffRouteEnergy(rd["dist_off"], rd["in_inter"], weight=W_NC,
                               margin=float(os.environ.get("ROUTE_MARGIN", "0.0")))]
    if ARM in ("corner", "mapcorner", "corneroffroute"):
        # footprint-corner drivable energy: the rectangle the scorer checks, not an eroded point
        from guidedwam.energies.navhard_fp import CornerDrivableEnergy
        d = np.load(FIELDS / f"ora_{token}.npz")
        ens = []
        if ARM == "mapcorner":
            ens.append(LaneCenterEnergy(np.load(FIELDS / f"lanes_{token}.npz")["pts"], weight=W_DAC))
        ens.append(CornerDrivableEnergy(d["dist_out"], weight=W_NC if ARM != "corneroffroute" else W_DAC,
                                        margin=float(os.environ.get("CORNER_MARGIN", "0.5"))))
        if ARM == "corneroffroute":
            # oracle footprint + DRIVING DIRECTION gate (plan §1); W_NC is the direction's relative weight
            from guidedwam.energies.navhard_fp import OffRouteEnergy
            rd = np.load(FIELDS / f"route_{token}.npz")
            ens.append(OffRouteEnergy(rd["dist_off"], rd["in_inter"], weight=W_NC,
                                      margin=float(os.environ.get("ROUTE_MARGIN", "0.0"))))
        return ens
    if ARM in ("drvfp", "boxdrvfp"):
        from guidedwam.energies.navhard_fp import FootprintDrivableEnergy
        d = np.load(FIELDS / f"ora_{token}.npz")
        ens = []
        if ARM == "boxdrvfp":
            ens.append(BoxClearanceEnergy(d["boxes"], d["vel"], weight=W_NC))
        ens.append(FootprintDrivableEnergy(
            d["dist_out"], weight=W_DAC,
            ego_half_width=float(os.environ.get("EGO_HALF_W", "1.0"))))
        return ens
    if ARM in ("box", "drivable", "boxdrivable"):
        d = np.load(FIELDS / f"ora_{token}.npz")
        ens = []
        if ARM in ("box", "boxdrivable"):
            ens.append(BoxClearanceEnergy(d["boxes"], d["vel"], weight=W_NC))
        if ARM in ("drivable", "boxdrivable"):
            ens.append(TrueDrivableEnergy(d["dist_out"], weight=W_DAC))
        return ens
    raise ValueError(f"unknown GUIDANCE_ARM={ARM}")


if sample is None:
    sample = [(t, "", "") for t in loader.tokens]
first = True
n_ok = n_fail = 0
for token, stage, gates in sample:
    if (OUT / f"{token}.npy").exists():
        n_ok += 1
        continue
    try:
        scene = loader.get_scene_from_token(token)
        cur = max(0, min(int(scene.scene_metadata.num_history_frames) - 1, len(scene.frames) - 1))
        img = scene.frames[cur].cameras.cam_f0.image
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        images = torch.as_tensor(np.stack([img] * num_frames), dtype=torch.uint8)
        images = images.permute(0, 3, 1, 2).contiguous().unsqueeze(0)
        hist = torch.as_tensor(scene.get_history_trajectory().poses, dtype=torch.float32)[-hist_dim:]
        if hist.shape[0] < hist_dim:
            hist = torch.cat([torch.zeros(hist_dim - hist.shape[0], hist.shape[1]), hist], 0)
        ego = scene.frames[cur].ego_status
        cmd = torch.as_tensor(ego.driving_command, dtype=torch.float32)
        command_id = int(torch.argmax(cmd).item())
        batch = {
            "images": images,
            "prompt": [prompts[min(command_id, len(prompts) - 1)]],
            "command_id": torch.tensor([command_id], dtype=torch.long),
            "ego_velocity_acc": torch.cat([
                torch.as_tensor(ego.ego_velocity, dtype=torch.float32),
                torch.as_tensor(ego.ego_acceleration, dtype=torch.float32)], 0).unsqueeze(0),
            "history_poses": hist.unsqueeze(0),
            "actions": ACTIONS,
            "actions_mask": MASK,
            "scene_token": [token],
        }
        if first:
            os.environ["DRIVEWAM_GUIDANCE_DEBUG"] = "1"
        _ens = with_speed_cap(build_energy(token))
        pred_actions, _ = evaluator.predict(
            batch, num_video_steps=3, num_action_steps=int(os.environ.get("NUM_ACTION_STEPS", "10")),
            seed=int(os.environ.get("PRED_SEED", "0")),   # vary to measure run-to-run spread
            video_denoise_fraction=0.6,
            energies=_ens,
            guidance_jacobian=bool(int(os.environ.get("JACOBIAN", "0"))),   # plan §5 DPS form
            guidance_clip_rel=float(os.environ.get("GUIDANCE_CLIP_REL", "12.0")),   # velocity bound c; 1e9 = off
            # ARM=off -> no energies and no schedule, so the guidance path is bypassed entirely.
            # WINDOW (ablation plan §4): asymmetric (trapezoid, default) | uniform | canonical | early |
            # late, calibrated to the trapezoid's total weight over this sampler's sigma grid
            # (action_snr_shift=1 -> linspace(1,0,n+1)[:-1]).
            guidance_schedule=GuidanceSchedule(mode=os.environ.get("WINDOW", "asymmetric"), lam_max=LAM).calibrate(
                np.linspace(1.0, 0.0, int(os.environ.get("NUM_ACTION_STEPS", "10")) + 1)[:-1]) if _ens else None)
        if first:
            os.environ.pop("DRIVEWAM_GUIDANCE_DEBUG", None)
            first = False
        poses = evaluator.pred_traj_to_poses(evaluator.actions_to_trajectory(pred_actions))
        np.save(OUT / f"{token}.npy", np.asarray(poses, dtype=np.float32))
        if os.environ.get("GUIDANCE_STATS"):
            from src.navsim import guidance as _g
            with open(OUT.parent / "guidance_stats.jsonl", "a") as fh:
                fh.write(json.dumps({"token": token, "steps": _g.STATS, "timing": _g.TIMING, "jac": _g.JAC_DIAG, "trace": _g.TRACE}) + "\n")
            _g.STATS.clear(); _g.TIMING.clear(); _g.JAC_DIAG.clear(); _g.TRACE.clear()
        n_ok += 1
        if n_ok % 200 == 0: print(f"{n_ok} preds", flush=True)
    except Exception as e:
        print(f"FAILED {token}: {type(e).__name__}: {e}", flush=True)
        n_fail += 1
print(f"arm={ARM} lam={LAM}: ok={n_ok} fail={n_fail} -> {OUT}")
