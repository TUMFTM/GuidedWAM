"""One panel per energy (drivable area on the learned field, lane keeping, driving direction, speed cap): the field
and the descent direction -dE/dtau on the waypoints of a stock plan, in the ego frame (x forward, y left).

Usage: fig_energy_gradients.py <pred_dir> <out.pdf> [token]   (fields venv; FIELDS_DIR and LEARNED_FIELDS from env.sh)
Without a token, picks the scene of data/scene_lists/pov_bev_cases.json where all four energies are active on the plan.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from guidedwam import paths
from guidedwam.energies.kinematic import SpeedCapEnergy, speeds_from_poses
from guidedwam.energies.navhard import LaneCenterEnergy
from guidedwam.energies.navhard_fp import LearnedCornerDrivableEnergy, OffRouteEnergy

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})
PRED, OUT = Path(sys.argv[1]), sys.argv[2]
FIELDS, LEARNED = Path(os.environ["FIELDS_DIR"]), Path(os.environ["LEARNED_FIELDS"])
CAP = float(os.environ.get("SPEED_CAP", "5.0"))


def energies(tok):
    sdf = np.load(LEARNED / f"bsdf_{tok}.npz")["sdf"].astype(np.float32)
    rd = np.load(FIELDS / f"route_{tok}.npz")
    return {"DA": LearnedCornerDrivableEnergy(sdf, margin=1.5), "LK": LaneCenterEnergy(np.load(FIELDS / f"lanes_{tok}.npz")["pts"]),
            "DD": OffRouteEnergy(rd["dist_off"], rd["in_inter"]), "cap": SpeedCapEnergy(CAP)}


def grads(ens, tau):
    out = {}
    for k, e in ens.items():
        x = torch.tensor(tau)[None].requires_grad_(True)
        E = e(x).sum()
        (g,) = torch.autograd.grad(E, x) if E.requires_grad else (torch.zeros_like(x),)
        out[k] = (float(E), -g[0, :, :2].numpy())
    return out


if len(sys.argv) > 3:
    TOK = sys.argv[3]
else:
    cands = [t for v in json.load(open(paths.REPO / "data/scene_lists/pov_bev_cases.json")).values() for t in v]
    best = None
    for t in cands:
        try:
            g = grads(energies(t), np.load(PRED / f"{t}.npy").astype(np.float32))
        except FileNotFoundError:
            continue
        score = min(g[k][0] > 1e-6 for k in g), sum(np.log1p(g[k][0]) for k in g)
        if best is None or score > best[0]:
            best = (score, t)
    TOK = best[1]
tau = np.load(PRED / f"{TOK}.npy").astype(np.float32)
ens = energies(TOK)
G = grads(ens, tau)
print("token", TOK, {k: round(v[0], 3) for k, v in G.items()})

fig, ax = plt.subplots(1, 4, figsize=(7.0, 2.6), constrained_layout=True)
pxy = np.vstack([[0, 0], tau[:, :2]])


def bev(a, field, xmin, ymin, cell, title, cmap, vmin, vmax, key):
    NX, NY = field.shape
    a.imshow(field, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
             extent=[ymin - cell / 2, ymin + (NY - 0.5) * cell, xmin - cell / 2, xmin + (NX - 0.5) * cell])
    a.plot(-pxy[:, 1], pxy[:, 0], "-o", color="k", ms=2.5, lw=1)   # plot axis: horizontal = -y (left is left)
    g = G[key][1]
    s = 3.0 / max(np.abs(g).max(), 1e-9)
    a.quiver(-tau[:, 1], tau[:, 0], -g[:, 1] * s, g[:, 0] * s, angles="xy", scale_units="xy", scale=1,
             color="tab:red", width=0.012)
    a.set_xlim(-16, 16); a.set_ylim(-4, 36); a.set_title(title); a.set_xticks([]); a.set_yticks([])


sdf = ens["DA"].field.numpy()
bev(ax[0], np.flip(sdf, 1), 0.0, -32.0, 0.25, "drivable area", "RdBu_r", -4, 4, "DA")
lanes = np.load(FIELDS / f"lanes_{TOK}.npz")["pts"]
ax[1].scatter(-lanes[:, 1], lanes[:, 0], s=0.3, color="0.6")
bev(ax[1], np.zeros((2, 2)), 0, 0, 0.25, "lane keeping", "Greys", 0, 1, "LK")
dd = ens["DD"].field.numpy()
bev(ax[2], np.flip(dd, 1), -10.0, -25.0, 0.25, "driving direction", "RdBu_r", -4, 4, "DD")
# the BEV extents above flip y so that left of the ego is left in the image; quiver/plot use -y for the same reason
v = speeds_from_poses(torch.tensor(tau)[None])[0].numpy()
t = 0.5 * np.arange(1, 9)
ax[3].plot(t, v, "-o", color="k", ms=2.5, lw=1, label="plan")
ax[3].axhline(CAP, color="tab:red", lw=1, ls="--", label=f"cap {CAP:g} m/s")
dv = -2.0 * np.maximum(v - CAP, 0.0)                     # -dE/dv_t of the hinge: pushes only speeds above the cap
ax[3].quiver(t, v, np.zeros_like(t), dv / max(np.abs(dv).max(), 1e-9) * 0.8 * max(v.max() - CAP, 0.5), color="tab:red",
             angles="xy", scale_units="xy", scale=1, width=0.012)
ax[3].set_xlabel("time [s]"); ax[3].set_ylabel("speed [m/s]"); ax[3].set_title("speed cap"); ax[3].legend(frameon=False, fontsize=6)
fig.savefig(OUT)
print("wrote", OUT)
