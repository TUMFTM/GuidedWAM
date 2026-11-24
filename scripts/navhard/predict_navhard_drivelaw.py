"""DriveLaW-Act trajectory prediction over NAVSIM v2 ``navhard_two_stage`` (+ optional energy guidance).

DriveLaW's own pipeline is a navsim-v1 fork (navtest only). This adapter drives the **v2** SceneLoader
(original stage-one + synthetic stage-two scenes), builds the agent's features with DriveLaW's own
``VideoDriveFeatureBuilder`` (last 4 CAM_F0 frames at 768x1344, driving command, velocity, acceleration)
and calls ``VideoDriveAgent.forward`` directly (the target builders need future frames, which synthetic
scenes do not have). Output: ``<token>.npy`` (8, 3) absolute poses @ 0.5 s in the current ego frame.

Guidance arms (env GUIDANCE_ARM, same as guided_navhard_drivewam.py): off | map | corner | mapcorner |
box | drvfp | boxdrvfp | mapdrvfp ...; GUIDANCE_LAM, FS_W_DAC, FS_W_NC, CORNER_MARGIN, EGO_HALF_W,
GUIDANCE_CLIP_REL. TOKENS_FILE restricts to a subset (search); ALL_TOKENS=1 runs the full split.

Run with the drivelaw venv after sourcing scripts/env.sh (PYTHONPATH = third_party/navsim first, then the repo).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from guidedwam import paths
import navsim
import navsim.agents
assert "third_party/navsim/" in str(navsim.__file__), f"navsim resolved to {navsim.__file__}"
# DriveLaW's agent package (navsim.agents.videodrive) lives in its own v1 devkit fork
navsim.agents.__path__.append(str(paths.DRIVELAW / "DriveLaW-Act/navsim/agents"))
from hydra.utils import instantiate
from omegaconf import OmegaConf
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.agents.videodrive.videodrive_agent import VideoDriveAgent
from navsim.agents.videodrive.videodrive_features import VideoDriveFeatureBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from guidedwam.energies.navhard import FreespaceFieldEnergy, LaneCenterEnergy, TrueDrivableEnergy, BoxClearanceEnergy
from guidedwam.energies.navhard_fp import FootprintDrivableEnergy, CornerDrivableEnergy
from guidedwam.sampler.drivelaw import GuidanceSchedule

ROOT = paths.NAVSIM_DATA
SIMWAM = paths.SIMWAM
FIELDS = Path(os.environ.get("FIELDS_DIR", str(paths.EXP / "navhard_fields")))
OUT = Path(os.environ.get("OUT_DIR", str(paths.EXP / "navhard_pred/drivelaw/pred_actions")))
OUT.mkdir(parents=True, exist_ok=True)
CFG = os.environ.get("DRIVELAW_CFG", str(paths.REPO / "configs/drivelaw_eval_navsim.yaml"))
_cfg_resolved = OUT.parent / "drivelaw_eval_navsim.resolved.yaml"   # the agent reads plain yaml: expand ${VAR}s here
_cfg_resolved.write_text(os.path.expandvars(Path(CFG).read_text())); CFG = str(_cfg_resolved)
ARM = os.environ.get("GUIDANCE_ARM", "off")
LAM = float(os.environ.get("GUIDANCE_LAM", "0.0"))
W_DAC = float(os.environ.get("FS_W_DAC", "1.0"))
W_NC = float(os.environ.get("FS_W_NC", "1.0"))
CLIP_REL = float(os.environ.get("GUIDANCE_CLIP_REL", "12.0"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "0"))


def build_energy(token):
    """List of energies for the selected arm (mirrors guided_navhard_drivewam.build_energy)."""
    if ARM == "off":
        return None
    if ARM == "speedcap":
        return []            # the speed cap alone; SPEED_CAP appends it to any arm
    if ARM == "freespace":
        d = np.load(FIELDS / f"fs_{token}.npz")
        return [FreespaceFieldEnergy(d["dist_free"], d["obst_pts"], w_dac=W_DAC, w_nc=W_NC)]
    if ARM == "map":
        return [LaneCenterEnergy(np.load(FIELDS / f"lanes_{token}.npz")["pts"], weight=W_DAC)]
    if ARM == "mapcorneroffroute":
        from guidedwam.energies.navhard_fp import OffRouteEnergy
        d = np.load(FIELDS / f"ora_{token}.npz"); rd = np.load(FIELDS / f"route_{token}.npz")
        return [LaneCenterEnergy(np.load(FIELDS / f"lanes_{token}.npz")["pts"], weight=W_DAC),
                CornerDrivableEnergy(d["dist_out"], weight=float(os.environ.get("CORNER_W", "1.0")),
                                     margin=float(os.environ.get("CORNER_MARGIN", "0.5"))),
                OffRouteEnergy(rd["dist_off"], rd["in_inter"], weight=W_NC,
                               margin=float(os.environ.get("ROUTE_MARGIN", "0.0")))]
    if ARM == "mapoffroute":
        from guidedwam.energies.navhard_fp import OffRouteEnergy
        rd = np.load(FIELDS / f"route_{token}.npz")
        return [LaneCenterEnergy(np.load(FIELDS / f"lanes_{token}.npz")["pts"], weight=W_DAC),
                OffRouteEnergy(rd["dist_off"], rd["in_inter"], weight=W_NC,
                               margin=float(os.environ.get("ROUTE_MARGIN", "0.0")))]
    if ARM in ("offroute", "corneroffroute", "cornerlearnoffroute"):
        # DRIVING DIRECTION gate: penalise distance travelled with the ego CENTRE outside the
        # on-route drivable polygons (pdm_scorer's "oncoming traffic"), intersections exempt.
        # Energy is in metres, ~100x the corner energy's scale, so LAM must be ~100x smaller.
        from guidedwam.energies.navhard_fp import OffRouteEnergy
        rd = np.load(FIELDS / f"route_{token}.npz")
        ens = []
        if ARM == "corneroffroute":
            d = np.load(FIELDS / f"ora_{token}.npz")
            ens.append(CornerDrivableEnergy(d["dist_out"], weight=W_DAC,
                                            margin=float(os.environ.get("CORNER_MARGIN", "0.5"))))
        if ARM == "cornerlearnoffroute":
            # compliant pairing: LEARNED corner field (margin default 1.5) + direction gate
            from guidedwam.energies.navhard_fp import LearnedCornerDrivableEnergy
            lf = Path(os.environ.get("LEARNED_FIELDS", str(paths.EXP / "bevsdf_fields_all")))
            d = np.load(lf / f"bsdf_{token}.npz")
            ens.append(LearnedCornerDrivableEnergy(d["sdf"].astype(np.float32), weight=W_DAC,
                                                   margin=float(os.environ.get("CORNER_MARGIN", "1.5"))))
        ens.append(OffRouteEnergy(rd["dist_off"], rd["in_inter"], weight=W_NC,
                                  margin=float(os.environ.get("ROUTE_MARGIN", "0.0"))))
        return ens
    if ARM == "cornerlearn":
        # corner energy over the TRAINED camera-only signed-distance field (bevsdf_train.py).
        # Compliant: no map at inference. Margin defaults to 1.5 — the learned field's false
        # positives hug the true edge, so a wide band buys recall cheaply.
        from guidedwam.energies.navhard_fp import LearnedCornerDrivableEnergy
        lf = Path(os.environ.get("LEARNED_FIELDS", str(paths.EXP / "bevsdf_fields_all")))
        d = np.load(lf / f"bsdf_{token}.npz")
        return [LearnedCornerDrivableEnergy(d["sdf"].astype(np.float32), weight=W_NC,
                                            margin=float(os.environ.get("CORNER_MARGIN", "1.5")))]
    if ARM in ("cornerfs", "mapcornerfs"):
        d = np.load(FIELDS / f"fs_{token}.npz")
        ens = []
        if ARM == "mapcornerfs":
            ens.append(LaneCenterEnergy(np.load(FIELDS / f"lanes_{token}.npz")["pts"], weight=W_DAC))
        ens.append(CornerDrivableEnergy(d["dist_free"], weight=W_NC, margin=float(os.environ.get("CORNER_MARGIN", "0.5"))))
        return ens
    if ARM in ("corner", "mapcorner", "boxcorner"):
        d = np.load(FIELDS / f"ora_{token}.npz")
        ens = []
        if ARM == "mapcorner":
            ens.append(LaneCenterEnergy(np.load(FIELDS / f"lanes_{token}.npz")["pts"], weight=W_DAC))
        if ARM == "boxcorner":
            ens.append(BoxClearanceEnergy(d["boxes"], d["vel"], weight=W_DAC))
        ens.append(CornerDrivableEnergy(d["dist_out"], weight=W_NC, margin=float(os.environ.get("CORNER_MARGIN", "0.5"))))
        return ens
    if ARM == "mapdrvfp":
        d = np.load(FIELDS / f"ora_{token}.npz")
        return [LaneCenterEnergy(np.load(FIELDS / f"lanes_{token}.npz")["pts"], weight=W_DAC),
                FootprintDrivableEnergy(d["dist_out"], weight=W_NC, ego_half_width=float(os.environ.get("EGO_HALF_W", "2.5")))]
    if ARM in ("drvfp", "boxdrvfp"):
        d = np.load(FIELDS / f"ora_{token}.npz")
        ens = []
        if ARM == "boxdrvfp":
            ens.append(BoxClearanceEnergy(d["boxes"], d["vel"], weight=W_NC))
        ens.append(FootprintDrivableEnergy(d["dist_out"], weight=W_DAC, ego_half_width=float(os.environ.get("EGO_HALF_W", "1.0"))))
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


SPLIT = os.environ.get("SPLIT", "navhard")          # navhard (two-stage, default) | navtest (v1 gate)
_sf = "navhard_two_stage.yaml" if SPLIT == "navhard" else "navtest.yaml"
scene_filter = instantiate(OmegaConf.load(
    paths.NAVSIM / "navsim/planning/script/config/common/train_test_split/scene_filter" / _sf))
_sensors = SensorConfig(cam_f0=True, cam_l0=False, cam_l1=False, cam_l2=False,
                        cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False)
if SPLIT == "navhard":
    assert getattr(scene_filter, "include_synthetic_scenes", False)
    loader = SceneLoader(
        data_path=ROOT / "navsim_logs/test", original_sensor_path=ROOT / "sensor_blobs/test",
        scene_filter=scene_filter,
        synthetic_sensor_path=ROOT / "navhard_two_stage/sensor_blobs",
        synthetic_scenes_path=ROOT / "navhard_two_stage/synthetic_scene_pickles",
        sensor_config=_sensors)
else:
    loader = SceneLoader(
        data_path=ROOT / "navsim_logs/test", original_sensor_path=ROOT / "sensor_blobs/test",
        scene_filter=scene_filter, synthetic_sensor_path=None, synthetic_scenes_path=None,
        sensor_config=_sensors)
tokens = sorted(loader.tokens)
if not os.environ.get("ALL_TOKENS"):
    tf = os.environ.get("TOKENS_FILE")
    if tf:
        keep = {t for t, *_ in json.load(open(tf))}
        tokens = [t for t in tokens if t in keep]
if MAX_TOKENS:
    tokens = tokens[:MAX_TOKENS]
syn = set(getattr(loader, "synthetic_scenes_tokens", []) or [])
print(f"{SPLIT} tokens: {len(tokens)} (synthetic in split: {len(syn)}) arm={ARM} lam={LAM}", flush=True)

agent = VideoDriveAgent(TrajectorySampling(time_horizon=4, interval_length=0.5), CFG, "", "cuda")
agent.initialize()
agent.eval()
agent.guidance_clip_rel = CLIP_REL
fb = VideoDriveFeatureBuilder(view_mode="front")

n_ok = n_fail = n_skip = 0
t_hist = []
for token in tqdm(tokens):
    out_path = OUT / f"{token}.npy"
    if out_path.exists():
        n_skip += 1
        continue
    try:
        agent_input = loader.get_agent_input_from_token(token)
        feats = fb.compute_features(agent_input)
        if "images" not in feats:
            raise ValueError("feature builder returned image paths, expected in-memory images")
        feats = {k: v.unsqueeze(0) for k, v in feats.items()}
        from guidedwam.energies.kinematic import with_speed_cap
        ens = with_speed_cap(build_energy(token))
        agent.guidance_energies = ens
        agent.guidance_schedule = GuidanceSchedule(lam_max=LAM) if (ens and (LAM > 0 or os.environ.get("GUIDANCE_TRACE"))) else None
        t0 = time.perf_counter()
        with torch.no_grad():
            poses = agent.forward(feats)["actions"]
        t_hist.append(time.perf_counter() - t0)
        poses = np.asarray(poses, dtype=np.float32)
        assert poses.shape == (8, 3), poses.shape
        np.save(out_path, poses)
        if os.environ.get("GUIDANCE_STATS"):
            import guidedwam.sampler.drivelaw as _g, json as _json
            with open(OUT.parent / "guidance_stats.jsonl", "a") as fh:
                fh.write(_json.dumps({"token": token, "steps": _g.STATS, "timing": _g.TIMING, "jac": _g.JAC_DIAG, "trace": _g.TRACE}) + "\n")
            _g.STATS.clear(); _g.TIMING.clear(); _g.JAC_DIAG.clear(); _g.TRACE.clear()
        n_ok += 1
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"FAILED {token}: {type(e).__name__}: {e}", flush=True)
        n_fail += 1
        if n_fail >= 5 and n_ok == 0:
            print("aborting: first 5 scenes all failed", flush=True); break

print(f"arm={ARM} lam={LAM}: ok={n_ok} fail={n_fail} skip={n_skip} -> {OUT}"
      + (f"  | {np.mean(t_hist):.2f} s/scene (median {np.median(t_hist):.2f})" if t_hist else ""), flush=True)
