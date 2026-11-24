"""Window-shape figure from the 40-scene instability search (x0_batch/): top = fraction of scenes whose final plan
zig-zags > 3x the stock plan; bottom = ADE and FDE of the final plan from the STOCK plan, uniform vs trapezoid.
Usage: fig_window_shape.py out.pdf"""
import glob, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
files = sorted(glob.glob("x0_batch/x0_*.npz")); lams = [float(v) for v in np.load(files[0])["lams"]]

def jitter(tr):
    y = np.vstack([[0, 0], tr[:, :2]]); d2 = y[2:] - 2 * y[1:-1] + y[:-2]; return float(np.sqrt((d2[:, 1] ** 2).mean()))

R = {k: {"fail": [], "ade": [], "fde": []} for k in ("const", "sched")}
for lam in lams:
    for k in R:
        f, a, e = [], [], []
        for fn in files:
            d = np.load(fn); g = d[f"{k}_final_{lam:g}"][:, :2]; s = d["off_final"][:, :2]
            f.append(jitter(d[f"{k}_final_{lam:g}"]) > 3 * max(jitter(d["off_final"]), 0.02))
            a.append(np.linalg.norm(g - s, axis=1).mean()); e.append(np.linalg.norm(g[-1] - s[-1]))
        R[k]["fail"].append(100 * np.mean(f)); R[k]["ade"].append(np.median(a)); R[k]["fde"].append(np.median(e))

C = {"const": "#d9480f", "sched": "#1c7ed6"}; L = {"const": "uniform", "sched": "trapezoid"}
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.3, 3.0), sharex=True, gridspec_kw={"hspace": 0.12})
for k in R:
    a1.plot(lams, R[k]["fail"], "-o", color=C[k], ms=3.5, lw=1.3, label=L[k])
    a2.plot(lams, R[k]["fde"], "-o", color=C[k], ms=3.5, lw=1.3)
    a2.plot(lams, R[k]["ade"], "--s", color=C[k], ms=3, lw=1.1)
for ax in (a1, a2):
    ax.set_xscale("log", base=2); ax.axvline(4, color="#868e96", lw=0.7, ls=":"); ax.grid(alpha=.25, lw=0.5)
a1.set_ylim(0, 100); a1.set_ylabel("failure rate [%]")
a1.text(4.15, 5, "operating\npoint", fontsize=6.5, color="#868e96", va="bottom")
a1.legend(loc="upper left", fontsize=7, frameon=False, bbox_to_anchor=(0.1, 1.0))
a2.set_yscale("log"); a2.set_ylabel("displacement error [m]")
a2.set_yticks([0.5, 1, 2, 5, 10, 25]); a2.set_yticklabels(["0.5", "1", "2", "5", "10", "25"]); a2.minorticks_off()
a2.set_xticks(lams); a2.set_xticklabels([f"{int(l)}" for l in lams]); a2.set_xlabel("guidance strength $\\lambda$")
a2.legend(handles=[Line2D([0], [0], color="k", marker="o", ms=3.5, lw=1.3, label="FDE"),
                   Line2D([0], [0], color="k", marker="s", ms=3, lw=1.1, ls="--", label="ADE")],
          loc="upper left", fontsize=7, frameon=False)
fig.savefig(sys.argv[1], bbox_inches="tight"); print("wrote", sys.argv[1])
for k in R: print(L[k], "fail", np.round(R[k]["fail"]), "fde", np.round(R[k]["fde"], 2), "ade", np.round(R[k]["ade"], 2))
