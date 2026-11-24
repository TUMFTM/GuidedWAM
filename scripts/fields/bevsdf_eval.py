"""Evaluate a trained BEV signed-distance net as a CORNER-ENERGY FIELD on navhard.

Runs the trained camera-only net over the DAC-failure scenes, then applies the same per-corner test
used for the released checkpoint and the camera freespace field (tfbev_corner_check.py), so the three
candidate fields are directly comparable:

  oracle DAC polygons      100.0% detection / 0.0% false alarm   (privileged, upper bound)
  released ltf_seed_0       16.9% / 1.3%                         (7-class semantic head)
  camera freespace          42.3% / 59.2%                        (SegFormer + DepthAnything)
  this net                  <- what we are measuring

Also reports field quality against the cached navhard SDF targets (MAE, MAE near the boundary,
sign accuracy, drivable IoU), split by stage, since stage two is 3DGS-rendered and stage one is not.

Env: CKPT (default bevsdf_run1/best.pt)  TOKENS_FILE  OUT  SAVE_FIELDS
Run with the fields venv after sourcing scripts/env.sh.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

import navsim
from hydra.utils import instantiate
from omegaconf import OmegaConf
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_model import TransfuserModel
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from guidedwam import paths

EXP = paths.EXP
ROOT = paths.NAVSIM_DATA
SIMWAM = paths.SIMWAM
FIELDS = EXP / "navhard_fields"
SDF_GT = EXP / "bevsdf_navhard"
PRED = EXP / "navhard_pred/drivewam/pred_actions"
CKPT = os.environ.get("CKPT", str(EXP / "bevsdf_run1/best.pt"))
OUT = Path(os.environ.get("OUT", EXP / "bevsdf_eval.json"))
SAVE_FIELDS = os.environ.get("SAVE_FIELDS")
TOKENS_FILE = os.environ.get("TOKENS_FILE", str(EXP / "navhard_fail_sample1200.json"))

CELL = 0.25
H, W = 128, 256
OXMIN, OYMIN, OCELL = -10.0, -25.0, 0.25
L, WD, RA2C = 5.176, 2.297, 1.461
CORN = np.array([[RA2C + L / 2, WD / 2], [RA2C + L / 2, -WD / 2],
                 [RA2C - L / 2, -WD / 2], [RA2C - L / 2, WD / 2]])
MARGINS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
DEPTHS = [0.25, 0.5, 1.0, 2.0]


def densify(poses, k=5):
    p = np.vstack([[0, 0, 0], poses]); out = []
    for a, b in zip(p[:-1], p[1:]):
        for s in np.linspace(0, 1, k, endpoint=False):
            out.append(a + s * (b - a))
    out.append(p[-1]); return np.array(out)


def corners(poses):
    c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
    x = poses[:, 0:1] + c[:, None] * CORN[:, 0][None] - s[:, None] * CORN[:, 1][None]
    y = poses[:, 1:2] + s[:, None] * CORN[:, 0][None] + c[:, None] * CORN[:, 1][None]
    return np.stack([x, y], -1).reshape(-1, 2)


def cam_feature(scene_cams):
    ims = []
    for name, crop in (("cam_l0", True), ("cam_f0", False), ("cam_r0", True)):
        im = getattr(scene_cams, name).image
        if im is None:
            return None
        ims.append(im[28:-28, 416:-416] if crop else im[28:-28])
    img = cv2.resize(np.concatenate(ims, axis=1), (1024, 256))
    return torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)


cfg = TransfuserConfig(); cfg.latent = True; cfg.num_bev_classes = 1
model = TransfuserModel(TrajectorySampling(time_horizon=4, interval_length=0.5), cfg)
ck = torch.load(CKPT, map_location="cpu")
model.load_state_dict(ck["model"])
model.eval().cuda()
print(f"loaded {CKPT} (epoch {ck.get('epoch')}, rec {ck.get('rec')})", flush=True)

scene_filter = instantiate(OmegaConf.load(
    paths.NAVSIM / "navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml"))
loader = SceneLoader(data_path=ROOT / "navsim_logs/test", original_sensor_path=ROOT / "sensor_blobs/test",
                     scene_filter=scene_filter,
                     synthetic_sensor_path=ROOT / "navhard_two_stage/sensor_blobs",
                     synthetic_scenes_path=ROOT / "navhard_two_stage/synthetic_scene_pickles",
                     sensor_config=SensorConfig(cam_f0=[3], cam_l0=[3], cam_l1=False, cam_l2=False,
                                                cam_r0=[3], cam_r1=False, cam_r2=False, cam_b0=False,
                                                lidar_pc=False))
syn = set(getattr(loader, "synthetic_scenes_tokens", []) or [])
tokens = [t for t, *_ in json.load(open(TOKENS_FILE))]
tokens = [t for t in tokens if t in set(loader.tokens)]
print(f"{len(tokens)} tokens ({sum(t in syn for t in tokens)} synthetic)", flush=True)
if SAVE_FIELDS:
    Path(SAVE_FIELDS).mkdir(parents=True, exist_ok=True)

rows = []
hit = np.zeros(len(MARGINS)); fa = np.zeros(len(MARGINS))
dep_n = np.zeros(len(DEPTHS) + 1); dep_hit = np.zeros(len(DEPTHS) + 1)
n_v = n_c = 0
n_fail = 0
for tok in tqdm(tokens):
    try:
        scene = loader.get_scene_from_token(tok)
        cur = scene.scene_metadata.num_history_frames - 1
        cam = cam_feature(scene.frames[cur].cameras)
        if cam is None:
            continue
        ego = scene.frames[cur].ego_status
        st = torch.from_numpy(np.concatenate([
            np.asarray(ego.driving_command, np.float32),
            np.asarray(ego.ego_velocity, np.float32),
            np.asarray(ego.ego_acceleration, np.float32)]))
        with torch.no_grad():
            sdf = model({"camera_feature": cam[None].cuda(),
                         "status_feature": st[None].cuda()})["bev_semantic_map"][0, 0].cpu().numpy()

        r = dict(tok=tok, stage="s2" if tok in syn else "s1")
        g = SDF_GT / f"sdf_{tok}.npz"
        if g.exists():
            gt = np.load(g)["sdf"].astype(np.float32)
            near = np.abs(gt) <= 1.0
            r["mae"] = float(np.abs(sdf - gt).mean())
            r["mae_near"] = float(np.abs(sdf - gt)[near].mean()) if near.any() else float("nan")
            r["sign_acc"] = float(((sdf < 0) == (gt < 0)).mean())
            i = ((sdf < 0) & (gt < 0)).sum(); u = ((sdf < 0) | (gt < 0)).sum()
            r["iou"] = float(i / max(u, 1))
            # by forward range: the average hides that 82% of first violations sit inside 16 m,
            # where the camera resolves the road edge far better than it does at 32 m
            xc = (np.arange(H) + 0.5) * CELL
            for lo, hi in ((0, 8), (8, 16), (16, 32)):
                band = (xc >= lo) & (xc < hi)
                bm = near[band]
                r[f"mae_near_{lo}_{hi}"] = (float(np.abs(sdf - gt)[band][bm].mean())
                                            if bm.any() else float("nan"))
                r[f"iou_{lo}_{hi}"] = float(((sdf < 0) & (gt < 0))[band].sum()
                                            / max(((sdf < 0) | (gt < 0))[band].sum(), 1))
        rows.append(r)
        if SAVE_FIELDS:
            np.savez_compressed(Path(SAVE_FIELDS) / f"bsdf_{tok}.npz", sdf=sdf.astype(np.float16))

        # ---- per-corner test against the oracle polygons
        p = PRED / f"{tok}.npy"
        if not p.exists():
            continue
        ora = np.load(FIELDS / f"ora_{tok}.npz")["dist_out"]
        cs = corners(densify(np.load(p).astype(float)))
        ix = np.clip(((cs[:, 0] - OXMIN) / OCELL).astype(int), 0, ora.shape[0] - 1)
        iy = np.clip(((cs[:, 1] - OYMIN) / OCELL).astype(int), 0, ora.shape[1] - 1)
        depth = ora[ix, iy]
        viol = depth > 1e-6
        if not viol.any():
            continue
        ci = (cs[:, 0] / CELL).astype(int); cj = (cs[:, 1] / CELL + 128).astype(int)
        ok = (ci >= 0) & (ci < H) & (cj >= 0) & (cj < W)
        sd_at = np.full(len(cs), np.nan)
        sd_at[ok] = sdf[ci[ok], cj[ok]]
        for b, mg in enumerate(MARGINS):
            fired = np.nan_to_num(sd_at, nan=-99.0) > -mg     # predicted sd + margin > 0
            hit[b] += (viol & fired).sum(); fa[b] += (~viol & fired).sum()
        fired0 = np.nan_to_num(sd_at, nan=-99.0) > 0
        for a, lo in enumerate([0.0] + DEPTHS):
            hi = ([0.0] + DEPTHS)[a + 1] if a + 1 <= len(DEPTHS) else np.inf
            m = viol & (depth >= lo) & (depth < hi)
            dep_n[a] += m.sum(); dep_hit[a] += (m & fired0).sum()
        n_v += viol.sum(); n_c += (~viol).sum()
    except Exception as e:
        n_fail += 1
        if n_fail <= 3:
            import traceback; traceback.print_exc()

json.dump(rows, open(OUT, "w"))


def m(key, sel=None):
    v = np.array([r[key] for r in rows if key in r and (sel is None or r["stage"] == sel)], float)
    v = v[np.isfinite(v)]
    return v.mean() if v.size else float("nan")


print(f"\nscenes {len(rows)} (fail {n_fail}) | violating corners {n_v}, compliant {n_c}")
print(f"\n{'field quality vs cached SDF target':<34} {'MAE':>7} {'MAE<1m':>8} {'sign':>7} {'IoU':>7}")
print("-" * 68)
for lab, sel in (("all failure scenes", None), ("stage one (real images)", "s1"), ("stage two (3DGS)", "s2")):
    print(f"{lab:<34} {m('mae',sel):>7.3f} {m('mae_near',sel):>8.3f} {m('sign_acc',sel):>7.4f} {m('iou',sel):>7.4f}")
print(f"\n{'boundary error by forward range (m from ego)':<44} {'MAE<1m':>9} {'IoU':>8}")
print("-" * 64)
for lo, hi in ((0, 8), (8, 16), (16, 32)):
    print(f"{f'  {lo}-{hi} m':<44} {m(f'mae_near_{lo}_{hi}'):>9.3f} {m(f'iou_{lo}_{hi}'):>8.4f}")
print(f"\n{'corner test, margin (m)':<26} {'detects violating':>18} {'false alarm':>13}")
print("-" * 60)
print(f"{'oracle polygons (ref)':<26} {'100.0%':>18} {'0.0%':>13}")
print(f"{'released ltf (ref)':<26} {'16.9%':>18} {'1.3%':>13}")
print(f"{'camera freespace (ref)':<26} {'42.3%':>18} {'59.2%':>13}")
for b, mg in enumerate(MARGINS):
    print(f"{('this net, margin %.2f' % mg):<26} {hit[b]/max(n_v,1):>17.1%} {fa[b]/max(n_c,1):>12.1%}")
print(f"\n{'depth outside (m)':<20} {'corners':>9} {'detected at margin 0':>21}")
print("-" * 54)
for a, lo in enumerate([0.0] + DEPTHS):
    hi = ([0.0] + DEPTHS)[a + 1] if a + 1 <= len(DEPTHS) else np.inf
    lab = f"{lo:.2f} - {hi:.2f}" if np.isfinite(hi) else f"> {lo:.2f}"
    if dep_n[a]:
        print(f"{lab:<20} {int(dep_n[a]):>9} {dep_hit[a]/dep_n[a]:>20.1%}")
