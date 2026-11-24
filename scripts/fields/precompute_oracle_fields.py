"""Precompute oracle fields for navhard: GT object boxes + true drivable area.

Per token writes ora_<token>.npz:
  boxes    (N,7) x,y,z,l,w,h,yaw   in the ego frame of the (perturbed) frame
  vel      (N,3) velocity_3d       same frame, for constant-velocity forecasting
  dist_out (NX,NY) float32         metres outside the drivable-area polygon union
                                   (0 inside), same BEV grid as the camera fields

Drivable area uses exactly the layers the PDM scorer's PDMDrivableMap builds from:
ROADBLOCK + ROADBLOCK_CONNECTOR + INTERSECTION + CARPARK_AREA — re-derived from
the map API + ego localization (not read from the metric cache).

Env: TOKENS_FILE (json list) or ALL_TOKENS=1; OUT_DIR.
"""
import json
import os
from pathlib import Path

import numpy as np
import shapely
from scipy.ndimage import distance_transform_edt
from shapely.geometry import MultiPolygon

import navsim
assert "third_party/navsim/" in str(navsim.__file__)
from hydra.utils import instantiate
from omegaconf import OmegaConf
from navsim.common.dataclasses import Scene, SensorConfig
from navsim.common.dataloader import SceneLoader
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import SemanticMapLayer
from guidedwam import paths

ROOT = paths.NAVSIM_DATA
SIMWAM = paths.SIMWAM
OUT = Path(os.environ.get("OUT_DIR", str(paths.EXP / "navhard_fields")))
OUT.mkdir(parents=True, exist_ok=True)
XMIN, XMAX, YMIN, YMAX, CELL = -10.0, 60.0, -25.0, 25.0, 0.25
NX, NY = int((XMAX - XMIN) / CELL), int((YMAX - YMIN) / CELL)
LAYERS = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR,
          SemanticMapLayer.INTERSECTION, SemanticMapLayer.CARPARK_AREA]
RADIUS = 80.0

scene_filter = instantiate(OmegaConf.load(
    paths.NAVSIM / "navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml"))
loader = SceneLoader(
    data_path=ROOT / "navsim_logs/test", original_sensor_path=ROOT / "sensor_blobs/test",
    scene_filter=scene_filter,
    synthetic_sensor_path=ROOT / "navhard_two_stage/sensor_blobs",
    synthetic_scenes_path=ROOT / "navhard_two_stage/synthetic_scene_pickles",
    sensor_config=SensorConfig.build_no_sensors(),
)
if os.environ.get("ALL_TOKENS"):
    tokens = list(loader.tokens)
else:
    tokens = [t for t, *_ in json.load(open(os.environ["TOKENS_FILE"]))]

# ego-frame grid cell centres, reused every scene
gx_cells = XMIN + (np.arange(NX) + 0.5) * CELL
gy_cells = YMIN + (np.arange(NY) + 0.5) * CELL
GX, GY = np.meshgrid(gx_cells, gy_cells, indexing="ij")
GXr, GYr = GX.ravel(), GY.ravel()

map_apis = {}
n_ok = n_skip = n_fail = 0
for i, token in enumerate(tokens):
    out = OUT / f"ora_{token}.npz"
    if out.exists():
        n_skip += 1
        continue
    try:
        scene = loader.get_scene_from_token(token)
        cur = max(0, min(int(scene.scene_metadata.num_history_frames) - 1, len(scene.frames) - 1))
        frame = scene.frames[cur]
        ann = frame.annotations
        boxes = np.asarray(ann.boxes, dtype=np.float32).reshape(-1, 7)
        vel = np.asarray(ann.velocity_3d, dtype=np.float32).reshape(-1, 3)

        ego = frame.ego_status
        gx, gy, gyaw = [float(v) for v in ego.ego_pose]
        map_api = scene.map_api
        if map_api is None:
            name = scene.scene_metadata.map_name
            if name not in map_apis:
                map_apis[name] = Scene._build_map_api(name)
            map_api = map_apis[name]

        objs = map_api.get_proximal_map_objects(Point2D(gx, gy), RADIUS, LAYERS)
        polys = []
        for layer in objs.values():
            for o in layer:
                p = getattr(o, "polygon", None)
                if p is not None and not p.is_empty:
                    polys.append(p)
        if polys:
            union = shapely.union_all(MultiPolygon([p for p in polys if p.geom_type == "Polygon"]
                                                   or polys).buffer(0))
            # grid cell centres (ego frame) -> global, then test containment
            c, s = np.cos(gyaw), np.sin(gyaw)
            px = gx + c * GXr - s * GYr
            py = gy + s * GXr + c * GYr
            inside = shapely.contains_xy(union, px, py).reshape(NX, NY)
        else:
            inside = np.zeros((NX, NY), bool)
        dist_out = (distance_transform_edt(~inside) * CELL).astype(np.float32) if inside.any() \
            else np.full((NX, NY), 50.0, np.float32)

        np.savez_compressed(out, boxes=boxes, vel=vel, dist_out=dist_out,
                            inside_frac=np.float32(inside.mean()))
        n_ok += 1
    except Exception as e:
        print(f"FAILED {token}: {type(e).__name__}: {e}", flush=True)
        n_fail += 1
    if (i + 1) % 200 == 0:
        print(f"{i+1}/{len(tokens)} ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)
print(f"oracle fields done: ok={n_ok} skip={n_skip} fail={n_fail} -> {OUT}")
