"""Training targets for the BEV signed-distance field: signed distance to the DAC drivable edge,
on the TransFuser BEV grid.

Per token writes sdf_<token>.npz with
  sdf (128, 256) float16   signed metres to the boundary of the scorer's drivable area,
                           NEGATIVE inside / POSITIVE outside, clipped to +-CAP.
                           Grid: [i, j], i = x/0.25 over x in [0, 32] m forward,
                                         j = y/0.25 + 128 over y in [-32, 32] m lateral
                           (identical convention to TransfuserConfig.bev_semantic_frame).

Layers are exactly the ones PDMDrivableMap builds from, i.e. what DAC is scored against —
NOT the LANE+INTERSECTION set the stock BEV semantic head is trained on.

Shards by LOG so each worker holds only its own scene dicts in RAM.
Env: SPLIT=navtrain|navhard  SHARD=i  NSHARD=n  OUT_DIR=...
"""
import os
from pathlib import Path

import numpy as np
import shapely
from omegaconf import OmegaConf
from scipy.ndimage import distance_transform_edt
from shapely.geometry import MultiPolygon

import navsim
from hydra.utils import instantiate
from navsim.common.dataclasses import Scene, SensorConfig
from navsim.common.dataloader import SceneLoader
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import SemanticMapLayer
from guidedwam import paths

ROOT = paths.NAVSIM_DATA
NAVSIM_V1 = paths.NAVSIM
SIMWAM = paths.SIMWAM
SPLIT = os.environ.get("SPLIT", "navtrain")
SHARD = int(os.environ.get("SHARD", "0"))
NSHARD = int(os.environ.get("NSHARD", "1"))
OUT = Path(os.environ.get("OUT_DIR", str(paths.EXP / f"bevsdf_{SPLIT}")))
OUT.mkdir(parents=True, exist_ok=True)
CAP = 8.0
CELL = 0.25
H, W = 128, 256
LAYERS = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR,
          SemanticMapLayer.INTERSECTION, SemanticMapLayer.CARPARK_AREA]
RADIUS = 70.0

# ego-frame cell centres, matching _coords_to_pixel: cell i spans x in [i*0.25, (i+1)*0.25)
XC = (np.arange(H) + 0.5) * CELL
YC = (np.arange(W) - 127.5) * CELL
GX, GY = np.meshgrid(XC, YC, indexing="ij")
GXr, GYr = GX.ravel(), GY.ravel()

if SPLIT == "navtrain":
    cfg = OmegaConf.load(NAVSIM_V1 / "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml")
    logs = sorted(list(cfg.log_names))
    # shard by log, but KEEP cfg.tokens: log_names and tokens are both applied as filters
    # (dataloader.filter_scenes), and dropping the token list pulls in every frame of each log,
    # about 5x the navtrain sample count.
    cfg.log_names = [l for k, l in enumerate(logs) if k % NSHARD == SHARD]
    scene_filter = instantiate(cfg)
    loader = SceneLoader(data_path=ROOT / "navsim_logs/trainval",
                         original_sensor_path=ROOT / "sensor_blobs/trainval",
                         scene_filter=scene_filter, synthetic_sensor_path=None,
                         synthetic_scenes_path=None, sensor_config=SensorConfig.build_no_sensors())
    tokens = sorted(loader.tokens)
else:
    scene_filter = instantiate(OmegaConf.load(
        paths.NAVSIM / "navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml"))
    loader = SceneLoader(data_path=ROOT / "navsim_logs/test", original_sensor_path=ROOT / "sensor_blobs/test",
                         scene_filter=scene_filter,
                         synthetic_sensor_path=ROOT / "navhard_two_stage/sensor_blobs",
                         synthetic_scenes_path=ROOT / "navhard_two_stage/synthetic_scene_pickles",
                         sensor_config=SensorConfig.build_no_sensors())
    tokens = sorted(loader.tokens)[SHARD::NSHARD]

print(f"[shard {SHARD}/{NSHARD}] {SPLIT}: {len(tokens)} tokens -> {OUT}", flush=True)

map_apis = {}
n_ok = n_skip = n_fail = 0
for i, token in enumerate(tokens):
    out = OUT / f"sdf_{token}.npz"
    if out.exists():
        n_skip += 1
        continue
    try:
        scene = loader.get_scene_from_token(token)
        cur = max(0, min(int(scene.scene_metadata.num_history_frames) - 1, len(scene.frames) - 1))
        gx, gy, gyaw = [float(v) for v in scene.frames[cur].ego_status.ego_pose]
        map_api = scene.map_api
        if map_api is None:
            name = scene.scene_metadata.map_name
            if name not in map_apis:
                map_apis[name] = Scene._build_map_api(name)
            map_api = map_apis[name]

        polys = []
        for layer in map_api.get_proximal_map_objects(Point2D(gx, gy), RADIUS, LAYERS).values():
            for o in layer:
                p = getattr(o, "polygon", None)
                if p is not None and not p.is_empty:
                    polys.append(p)
        if polys:
            union = shapely.union_all(MultiPolygon([p for p in polys if p.geom_type == "Polygon"]
                                                   or polys).buffer(0))
            c, s = np.cos(gyaw), np.sin(gyaw)
            inside = shapely.contains_xy(union, gx + c * GXr - s * GYr,
                                         gy + s * GXr + c * GYr).reshape(H, W)
        else:
            inside = np.zeros((H, W), bool)

        d_out = distance_transform_edt(~inside) * CELL if (~inside).any() else np.zeros((H, W))
        d_in = distance_transform_edt(inside) * CELL if inside.any() else np.zeros((H, W))
        sdf = np.clip(d_out - d_in, -CAP, CAP).astype(np.float16)
        np.savez_compressed(out, sdf=sdf, inside_frac=np.float32(inside.mean()))
        n_ok += 1
    except Exception as e:
        n_fail += 1
        if n_fail <= 3:
            print(f"FAILED {token}: {type(e).__name__}: {e}", flush=True)
    if (i + 1) % 2000 == 0:
        print(f"[shard {SHARD}] {i+1}/{len(tokens)} ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)
print(f"[shard {SHARD}] DONE ok={n_ok} skip={n_skip} fail={n_fail} -> {OUT}", flush=True)
