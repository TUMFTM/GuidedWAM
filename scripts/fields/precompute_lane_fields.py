"""Precompute forward-oriented lane-centerline points (local frame) for the
120 navhard joint-failure tokens — inputs for LaneCenterEnergy (oracle arm).

Per token: query LANE + LANE_CONNECTOR baselines within 60 m of the perturbed
global ego pose, transform to the local ego frame, keep points ahead-ish
(x > -10) whose lane direction is within +-80 deg of ego heading (forward
lanes only -> pulls plans out of oncoming lanes, targeting DDC as well as LK).

Output: lanes_<token>.npz {pts: (P,3) x,y,heading_local} in the freespace dir.
"""
import json
import os
from pathlib import Path

import numpy as np

import navsim
assert "third_party/navsim/" in str(navsim.__file__)
from hydra.utils import instantiate
from omegaconf import OmegaConf
from navsim.common.dataclasses import SensorConfig, Scene
from navsim.common.dataloader import SceneLoader
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import SemanticMapLayer
from guidedwam import paths

ROOT = paths.NAVSIM_DATA
SIMWAM = paths.SIMWAM
OUT = Path(os.environ.get("LANE_OUT", str(paths.EXP / "navhard_freespace120")))

if os.environ.get("ALL_TOKENS"):
    sample = None  # filled after loader init
else:
    sample = json.load(open(str(paths.EXP / "navhard_fail_sample120.json")))
scene_filter = instantiate(OmegaConf.load(
    paths.NAVSIM / "navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml"))
loader = SceneLoader(
    data_path=ROOT / "navsim_logs/test", original_sensor_path=ROOT / "sensor_blobs/test",
    scene_filter=scene_filter,
    synthetic_sensor_path=ROOT / "navhard_two_stage/sensor_blobs",
    synthetic_scenes_path=ROOT / "navhard_two_stage/synthetic_scene_pickles",
    sensor_config=SensorConfig.build_no_sensors(),
)

if sample is None:
    sample = [(t, "", "") for t in loader.tokens]
map_apis = {}
n_ok = 0
for token, stage, gates in sample:
    if (OUT / f"lanes_{token}.npz").exists():
        n_ok += 1
        continue
    scene = loader.get_scene_from_token(token)
    cur = max(0, min(int(scene.scene_metadata.num_history_frames) - 1, len(scene.frames) - 1))
    ego = scene.frames[cur].ego_status
    assert ego.in_global_frame
    gx, gy, gyaw = [float(v) for v in ego.ego_pose]
    map_api = scene.map_api
    if map_api is None:
        name = scene.scene_metadata.map_name
        if name not in map_apis:
            map_apis[name] = Scene._build_map_api(name)
        map_api = map_apis[name]

    objs = map_api.get_proximal_map_objects(
        Point2D(gx, gy), 60.0, [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR])
    c, s = np.cos(-gyaw), np.sin(-gyaw)
    pts = []
    for layer in objs.values():
        for lane in layer:
            try:
                path = lane.baseline_path.discrete_path
            except Exception:
                continue
            for p in path[::2]:
                lx = c * (p.x - gx) - s * (p.y - gy)
                ly = s * (p.x - gx) + c * (p.y - gy)
                lh = (p.heading - gyaw + np.pi) % (2 * np.pi) - np.pi
                if -10.0 < lx < 70.0 and abs(ly) < 30.0 and abs(lh) < np.deg2rad(80):
                    pts.append((lx, ly, lh))
    pts = np.asarray(pts, dtype=np.float32) if pts else np.zeros((0, 3), np.float32)
    np.savez_compressed(OUT / f"lanes_{token}.npz", pts=pts)
    n_ok += 1
    if n_ok % 200 == 0: print(f"{n_ok} lane fields", flush=True)
print(f"done {n_ok}")
