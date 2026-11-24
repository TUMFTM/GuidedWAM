"""Field for the DRIVING DIRECTION metric (the second energy).

`_calculate_driving_direction_compliance` scores distance travelled by the ego CENTRE while
"in oncoming traffic", which pdm_scorer defines exactly as:

    center NOT inside any drivable polygon that is ON-ROUTE          (pdm_scorer.py:359-361)

with progress zeroed inside INTERSECTION polygons, summed over a rolling 1 s window, then
thresholded: < 2 m -> 1.0, < 6 m -> 0.5, else 0.0.

So the field the energy needs is distance outside the union of ON-ROUTE drivable polygons, plus an
intersection mask to suppress it where the metric does. Both come from the metric cache, which is
what the scorer itself reads, so this reproduces the scored quantity rather than approximating it.

Per token writes route_<token>.npz:
  dist_off  (NX,NY) float32  metres outside the on-route union (0 inside)
  in_inter  (NX,NY) bool     inside an INTERSECTION polygon (metric ignores off-route progress here)

Env: TOKENS_FILE (json) or ALL_TOKENS=1; OUT_DIR; SHARD/NSHARD.
"""
import json
import os
from pathlib import Path

import numpy as np
import shapely
from scipy.ndimage import distance_transform_edt
from shapely.geometry import MultiPolygon

from navsim.common.dataloader import MetricCacheLoader
from nuplan.common.maps.abstract_map import SemanticMapLayer
from guidedwam import paths

CACHE = paths.METRIC_CACHE
OUT = Path(os.environ.get("OUT_DIR", str(paths.EXP / "navhard_fields")))
OUT.mkdir(parents=True, exist_ok=True)
SHARD, NSHARD = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARD", "1"))
XMIN, XMAX, YMIN, YMAX, CELL = -10.0, 60.0, -25.0, 25.0, 0.25
NX, NY = int((XMAX - XMIN) / CELL), int((YMAX - YMIN) / CELL)
gx_c = XMIN + (np.arange(NX) + 0.5) * CELL
gy_c = YMIN + (np.arange(NY) + 0.5) * CELL
GX, GY = np.meshgrid(gx_c, gy_c, indexing="ij")
GXr, GYr = GX.ravel(), GY.ravel()

loader = MetricCacheLoader(CACHE)
tokens = ([t for t, *_ in json.load(open(os.environ["TOKENS_FILE"]))]
          if os.environ.get("TOKENS_FILE") else sorted(loader.tokens))
tokens = tokens[SHARD::NSHARD]
print(f"[shard {SHARD}/{NSHARD}] {len(tokens)} tokens -> {OUT}", flush=True)

n_ok = n_skip = n_fail = 0
for i, tok in enumerate(tokens):
    out = OUT / f"route_{tok}.npz"
    if out.exists():
        n_skip += 1
        continue
    try:
        mc = loader.get_from_token(tok)
        dm = mc.drivable_area_map
        route = set(mc.route_lane_ids)
        ra = mc.ego_state.rear_axle
        gxp, gyp, gyaw = ra.x, ra.y, ra.heading
        c, s = np.cos(gyaw), np.sin(gyaw)
        px, py = gxp + c * GXr - s * GYr, gyp + s * GXr + c * GYr   # grid centres -> global

        on_route, inter = [], []
        for idx, tk in enumerate(dm.tokens):
            g = dm._geometries[idx]
            if g is None or g.is_empty:
                continue
            if tk in route:
                on_route.append(g)
            if dm.map_types[idx] == SemanticMapLayer.INTERSECTION:
                inter.append(g)

        if on_route:
            u = shapely.union_all(MultiPolygon([g for g in on_route if g.geom_type == "Polygon"]
                                               or on_route).buffer(0))
            inside = shapely.contains_xy(u, px, py).reshape(NX, NY)
        else:
            inside = np.zeros((NX, NY), bool)
        dist_off = (distance_transform_edt(~inside) * CELL).astype(np.float32) if inside.any() \
            else np.full((NX, NY), 50.0, np.float32)

        if inter:
            ui = shapely.union_all(MultiPolygon([g for g in inter if g.geom_type == "Polygon"]
                                                or inter).buffer(0))
            in_inter = shapely.contains_xy(ui, px, py).reshape(NX, NY)
        else:
            in_inter = np.zeros((NX, NY), bool)

        np.savez_compressed(out, dist_off=dist_off, in_inter=in_inter,
                            on_route_frac=np.float32(inside.mean()))
        n_ok += 1
    except Exception as e:
        n_fail += 1
        if n_fail <= 3:
            print(f"FAILED {tok}: {type(e).__name__}: {e}", flush=True)
    if (i + 1) % 500 == 0:
        print(f"[shard {SHARD}] {i+1}/{len(tokens)} ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)
print(f"[shard {SHARD}] DONE ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)
