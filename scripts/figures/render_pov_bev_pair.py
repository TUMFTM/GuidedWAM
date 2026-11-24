"""Side-by-side figure: front-camera POV (left) and BEV with the energy field (right).

POV: the three trajectories ray-cast onto the ground plane and drawn over the raw camera image, with
no field overlay, so the image stays readable.
BEV: the corner energy's own field as a colour layer — signed distance to the drivable edge plus the
energy's margin, so zero is exactly the threshold where the barrier starts pushing — with a colourbar,
plus lane centrelines, other agents, ego, and the same three trajectories.

Usage: render_pov_bev_pair.py <token> <out.png>
Env: GUIDED_DIR, CORNER_MARGIN (default 0.5, matching the `corner` arm), HALF_W, FWD, CAM
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
from matplotlib.patches import Polygon as MplPolygon
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely.geometry import MultiPolygon
import shapely

import navsim
from hydra.utils import instantiate
from omegaconf import OmegaConf
from navsim.common.dataclasses import Scene, SensorConfig
from navsim.common.dataloader import SceneLoader
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import SemanticMapLayer
from guidedwam import paths

TOK, OUTP = sys.argv[1], sys.argv[2]
EXP = paths.EXP
ROOT = paths.NAVSIM_DATA
SIMWAM = paths.SIMWAM
L, W, RA2C = 5.176, 2.297, 1.461
CORN = np.array([[RA2C + L / 2, W / 2], [RA2C + L / 2, -W / 2],
                 [RA2C - L / 2, -W / 2], [RA2C - L / 2, W / 2]])
DRIVABLE = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR,
            SemanticMapLayer.INTERSECTION, SemanticMapLayer.CARPARK_AREA]
XMIN, XMAX, YMIN, YMAX, CELL = -10.0, 60.0, -25.0, 25.0, 0.25
GROUND_Z = -0.357
MARGIN = float(os.environ.get("CORNER_MARGIN", "0.5"))
CAM = os.environ.get("CAM", "cam_f0")
ARMS = [("stock", Path(os.environ.get("STOCK_DIR", str(EXP / "navhard_pred/drivewam/pred_actions"))), "#adb5bd"),
        ("energy guidance", Path(os.environ.get("GUIDED_DIR",
         str(EXP / "navhard_validate/val_dwam_corner_l4.00/pred_actions"))), "#e8590c"),
        ("post-hoc projection", Path(os.environ.get("PROJ_DIR", str(EXP / "navhard_proj/dwam_stock_proj/pred_actions"))), "#1c7ed6")]


def to_local(xy, pose):
    gx, gy, gyaw = pose
    c, s = np.cos(-gyaw), np.sin(-gyaw)
    d = np.asarray(xy) - np.array([gx, gy])
    return np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], 1)


def footprint(p):
    c, s = np.cos(p[2]), np.sin(p[2])
    return np.array([[p[0] + c * cx - s * cy, p[1] + s * cx + c * cy] for cx, cy in CORN])


def project(xyz, K, R, t):
    uvw = ((xyz - t[None]) @ np.linalg.inv(R).T) @ K.T
    return uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-3), uvw[:, 2] > 1e-3


sf = instantiate(OmegaConf.load(
    paths.NAVSIM / "navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml"))
loader = SceneLoader(data_path=ROOT / "navsim_logs/test", original_sensor_path=ROOT / "sensor_blobs/test",
                     scene_filter=sf, synthetic_sensor_path=ROOT / "navhard_two_stage/sensor_blobs",
                     synthetic_scenes_path=ROOT / "navhard_two_stage/synthetic_scene_pickles",
                     sensor_config=SensorConfig(cam_f0=[3], cam_l0=[3], cam_l1=False, cam_l2=False,
                                                cam_r0=[3], cam_r1=False, cam_r2=False, cam_b0=False,
                                                lidar_pc=False))
scene = loader.get_scene_from_token(TOK)
cur = max(0, min(int(scene.scene_metadata.num_history_frames) - 1, len(scene.frames) - 1))
frame = scene.frames[cur]
pose = [float(v) for v in frame.ego_status.ego_pose]
map_api = scene.map_api or Scene._build_map_api(scene.scene_metadata.map_name)
trajs = {n: np.load(d / f"{TOK}.npy").astype(float) for n, d, _ in ARMS if (d / f"{TOK}.npy").exists()}
cols = {n: c for n, _, c in ARMS}

# ---------- the energy's own field: signed distance to the drivable edge, + margin ----------
ora = np.load(Path(os.environ.get("FIELDS_DIR", EXP / "navhard_fields")) / f"ora_{TOK}.npz")
dist_out = ora["dist_out"]
inside = dist_out <= 1e-6
d_in = distance_transform_edt(inside) * CELL if inside.any() else np.zeros(inside.shape)
signed = gaussian_filter(np.clip(dist_out - d_in, -8, 8).astype(np.float32), 1.0)
band = signed + MARGIN                      # >0 : the barrier pushes here

fig = plt.figure(figsize=(19.0, 6.4), facecolor="white")
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.08, 0.03], wspace=0.05,
                      left=.008, right=.95, top=.93, bottom=.03)

# ---------------- POV (no field overlay) ----------------
axp = fig.add_subplot(gs[0])
cam = getattr(frame.cameras, CAM)
img = cam.image
K = np.asarray(cam.intrinsics, float); R = np.asarray(cam.sensor2lidar_rotation, float)
t = np.asarray(cam.sensor2lidar_translation, float)
axp.imshow(img)
H, Wd = img.shape[:2]
for n, tr in trajs.items():
    pts = np.vstack([[0, 0, 0], tr])
    uv, ok = project(np.column_stack([pts[:, :2], np.full(len(pts), GROUND_Z)]), K, R, t)
    uv = uv.copy(); uv[(~ok) | (np.abs(uv) > 1e4).any(1)] = np.nan
    axp.plot(uv[:, 0], uv[:, 1], "--" if n == "stock" else "-", color=cols[n], lw=3.0, alpha=.95)
    axp.plot(uv[1:, 0], uv[1:, 1], "o", color=cols[n], ms=8, markeredgecolor="white",
             markeredgewidth=1.0, label=n)
    q, okq = project(np.column_stack([footprint(tr[-1]), np.full(4, GROUND_Z)]), K, R, t)
    if okq.all() and np.all(np.abs(q) < 1e4):
        axp.add_patch(MplPolygon(q, closed=True, fill=False, edgecolor=cols[n], lw=2.0))
axp.set_xlim(0, Wd); axp.set_ylim(H, 0); axp.axis("off")
axp.set_title("Front camera", fontsize=12)
axp.legend(loc="lower left", fontsize=11, framealpha=.92)

# ---------------- BEV with the energy field (landscape: forward = right, left-of-ego = up) --------
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
axb = fig.add_subplot(gs[1])
# field is [x forward, y lateral]; transpose so rows = lateral (vertical), cols = forward (horizontal)
im = axb.imshow(band.T, origin="lower", extent=[XMIN, XMAX, YMIN, YMAX],
                cmap="RdBu_r", vmin=-4, vmax=4, aspect="equal", zorder=1)

# exact map geometry for the lines (a contour of the 0.25 m raster comes out as a wavy staircase)
objs = map_api.get_proximal_map_objects(Point2D(pose[0], pose[1]), 80.0, DRIVABLE)
polys = [o.polygon for lay in objs.values() for o in lay if getattr(o, "polygon", None) is not None]
if polys:
    u = shapely.union_all(MultiPolygon([q for q in polys if q.geom_type == "Polygon"] or polys).buffer(0))
    for g, ls, lw in ((u, "-", 1.4), (u.buffer(-MARGIN), "--", 1.0)):
        for gg in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
            if gg.is_empty:
                continue
            for ring in [gg.exterior] + list(gg.interiors):
                xy = to_local(np.array(ring.coords), pose)
                axb.plot(xy[:, 0], xy[:, 1], ls, color="k", lw=lw, zorder=2)

lanes = map_api.get_proximal_map_objects(Point2D(pose[0], pose[1]), 80.0,
                                         [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR])
for lay in lanes.values():
    for o in lay:
        try:
            pts = np.array([[p.x, p.y] for p in o.baseline_path.discrete_path])
        except Exception:
            continue
        xy = to_local(pts, pose)
        axb.plot(xy[:, 0], xy[:, 1], "--", color="#495057", lw=0.9, alpha=.75, zorder=3)

for b in np.asarray(ora["boxes"], float).reshape(-1, 7):
    c, s_ = np.cos(b[6]), np.sin(b[6])
    hl, hw = b[3] / 2, b[4] / 2
    q = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    xy = np.stack([b[0] + c * q[:, 0] - s_ * q[:, 1], b[1] + s_ * q[:, 0] + c * q[:, 1]], 1)
    axb.add_patch(MplPolygon(xy, closed=True, facecolor="#343a40", edgecolor="black",
                             lw=0.7, alpha=.85, zorder=4))

for n, tr in trajs.items():
    pts = np.vstack([[0, 0, 0], tr])
    axb.plot(pts[:, 0], pts[:, 1], "--" if n == "stock" else "-", color=cols[n], lw=2.6, zorder=5)
    axb.plot(pts[1:, 0], pts[1:, 1], "o", color=cols[n], ms=7, markeredgecolor="white",
             markeredgewidth=.8, zorder=6)
    axb.add_patch(MplPolygon(footprint(tr[-1]), closed=True, fill=False, edgecolor=cols[n],
                             lw=1.8, zorder=6))
axb.add_patch(MplPolygon(footprint(np.array([0.0, 0.0, 0.0])), closed=True, facecolor="#f76707",
                         edgecolor="#d9480f", lw=1.2, zorder=7))

lim = float(os.environ.get("HALF_W", "14")); fwd = float(os.environ.get("FWD", "46"))
axb.set_xlim(-5, fwd); axb.set_ylim(-lim, lim)
axb.set_aspect("equal"); axb.set_xticks([]); axb.set_yticks([])
for sp in axb.spines.values():
    sp.set_visible(False)
axb.set_title("BEV with energy field", fontsize=12)
handles = [Line2D([0], [0], color="k", lw=1.4, label="drivable edge"),
           Line2D([0], [0], color="k", lw=1.0, ls="--", label=f"energy activation margin ({MARGIN:g} m)"),
           Patch(facecolor="#f76707", edgecolor="#d9480f", label="ego vehicle"),
           Patch(facecolor="#343a40", edgecolor="black", label="other vehicles")]
axb.legend(handles=handles, loc="lower right", fontsize=9, framealpha=.92)

cax = fig.add_subplot(gs[2])
# match the colourbar's height to the BEV axes (the gridspec cell is taller than the equal-aspect map)
fig.canvas.draw()
bb = axb.get_position()
cb_bb = cax.get_position()
cax.set_position([cb_bb.x0, bb.y0, cb_bb.width, bb.height])
cb = fig.colorbar(im, cax=cax)
cb.set_label("drivable area energy field [m]", fontsize=10)
cb.ax.axhline(0.0, color="k", lw=1.2)
fig.savefig(OUTP, dpi=180)  # pdf: lines/text vector, camera image + field raster
print(f"wrote {OUTP}")
