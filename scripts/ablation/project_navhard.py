"""Post-hoc projection baseline.

Projects saved navhard trajectories onto the SAME footprint-corner constraint the corner energy uses
(CornerDrivableEnergy, margin 0.5 m, 5.18 x 2.30 m rectangle, four corners on the signed-distance
field). Gradient projection by penalty method: minimise mu * E_corner(tau0 + delta) + ||delta||^2
over delta on all three channels (x, y, heading), Adam lr 0.01, <= 1000 iters, mu in {1, 10, 100, 1000}
escalating whenever the violation stops decreasing; stop when every corner satisfies sd + margin <= 0.
Scenes with no violation pass through unchanged; scenes still violating at max iters keep the
ORIGINAL trajectory (recorded as failed). Each plan is projected independently (no memory).

Usage: project_navhard.py <pred_dir> <out_dir> [fields_dir]   -> out_dir/<token>.npy + out_dir/_proj_stats.json
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch

from guidedwam.energies.navhard_fp import CornerDrivableEnergy
from guidedwam import paths

pred_dir, out_dir = sys.argv[1], sys.argv[2]
fields = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("FIELDS_DIR", str(paths.EXP / "navhard_fields"))
os.makedirs(out_dir, exist_ok=True)
MARGIN, LR, MAX_IT = 0.5, 0.01, 1000
MUS = [1.0, 10.0, 100.0, 1000.0]
SMOOTH = float(os.environ.get("PROJ_SMOOTH", "0"))   # weight of the second-difference penalty (0 = paper projector)


def second_diff(tau):
    """Sum of squared second differences of the waypoints, the fixed current pose (origin) prepended."""
    xy = torch.cat([torch.zeros_like(tau[:, :1, :2]), tau[:, :, :2]], dim=1)
    return ((xy[:, 2:] - 2 * xy[:, 1:-1] + xy[:, :-2]) ** 2).sum()
torch.set_num_threads(4)


def max_violation(E, tau):
    """Max (sd + margin) over the FUTURE footprint corners. The densified path's first point is the
    current pose at the origin, which the projector cannot move — it is excluded from the stop test
    (the energy still contains it as a constant term)."""
    X, Y = E.corner_xy(tau)
    return float((E._bilinear(X, Y)[:, 1:, :] + E.margin).max())


files = sorted(glob.glob(os.path.join(pred_dir, "*.npy")))
stats = {"n": 0, "active": 0, "failed": 0, "missing_field": 0, "disp": [], "iters": [], "ms": []}
per_scene = {}
for f in files:
    tok = os.path.basename(f)[:-4]
    tau0 = np.load(f).astype(np.float32)
    ff = os.path.join(fields, f"ora_{tok}.npz")
    if not os.path.exists(ff):
        stats["missing_field"] += 1
        np.save(os.path.join(out_dir, f"{tok}.npy"), tau0); continue
    stats["n"] += 1
    t0 = time.perf_counter()
    E = CornerDrivableEnergy(np.load(ff)["dist_out"], margin=MARGIN)
    T0 = torch.tensor(tau0)[None]
    v0 = max_violation(E, T0)
    if v0 <= 0.0:
        np.save(os.path.join(out_dir, f"{tok}.npy"), tau0)
        per_scene[tok] = {"active": False}
        stats["ms"].append(1000 * (time.perf_counter() - t0)); continue
    stats["active"] += 1
    delta = torch.zeros_like(T0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=LR)
    mu_i, best_v, stall, it, ok = 0, v0, 0, 0, False
    for it in range(1, MAX_IT + 1):
        opt.zero_grad()
        loss = MUS[mu_i] * E(T0 + delta).sum() + (delta ** 2).sum()
        if SMOOTH:
            loss = loss + SMOOTH * second_diff(T0 + delta)
        loss.backward(); opt.step()
        with torch.no_grad():
            v = max_violation(E, T0 + delta)
        if v <= 0.0:
            ok = True; break
        if v < best_v - 1e-4:
            best_v, stall = v, 0
        else:
            stall += 1
            if stall >= 5 and mu_i < len(MUS) - 1:
                mu_i += 1; stall = 0
    ms = 1000 * (time.perf_counter() - t0)
    stats["ms"].append(ms); stats["iters"].append(it)
    if ok:
        tau = (T0 + delta).detach()[0].numpy().astype(np.float32)
        disp = float(np.linalg.norm(tau[:, :2] - tau0[:, :2], axis=1).max())
        stats["disp"].append(disp)
        np.save(os.path.join(out_dir, f"{tok}.npy"), tau)
        per_scene[tok] = {"active": True, "ok": True, "disp": disp, "iters": it, "v0": v0,
                          "dheading_max": float(np.abs(tau[:, 2] - tau0[:, 2]).max())}
    else:
        stats["failed"] += 1
        np.save(os.path.join(out_dir, f"{tok}.npy"), tau0)
        per_scene[tok] = {"active": True, "ok": False, "iters": it, "v0": v0, "v_end": float(best_v)}

d = np.array(stats["disp"]) if stats["disp"] else np.zeros(1)
summary = {
    "pred_dir": pred_dir, "out_dir": out_dir, "n": stats["n"], "missing_field": stats["missing_field"],
    "activation_rate": stats["active"] / max(stats["n"], 1), "failure_rate_of_active": stats["failed"] / max(stats["active"], 1),
    "failed": stats["failed"], "disp_mean_m": float(d.mean()), "disp_p90_m": float(np.percentile(d, 90)), "disp_max_m": float(d.max()),
    "iters_mean": float(np.mean(stats["iters"])) if stats["iters"] else 0.0,
    "ms_per_scene_all": float(np.mean(stats["ms"])), "ms_per_scene_active": float(np.mean([m for m, i in zip(stats["ms"], range(len(stats["ms"])))])),
    "margin": MARGIN, "lr": LR, "max_iters": MAX_IT, "mus": MUS, "smooth": SMOOTH,
}
json.dump({"summary": summary, "per_scene": per_scene}, open(os.path.join(out_dir, "_proj_stats.json"), "w"))
print(f"PROJ {pred_dir}: n={summary['n']} active={summary['activation_rate']*100:.1f}% failed={stats['failed']} "
      f"disp mean={summary['disp_mean_m']:.2f} p90={summary['disp_p90_m']:.2f} max={summary['disp_max_m']:.2f} m | "
      f"iters mean={summary['iters_mean']:.0f} | {summary['ms_per_scene_all']:.1f} ms/scene", flush=True)
