"""Cross-fitted selection + paired bootstrap over the 225 pseudo-closed-loop groups (plan §2).

Inputs: per-group score files from group_scores.py (groups_<run>.npz). Groups are split into two folds
by a hash of the group id; the arm with the best fold-A mean is selected and reported on fold B, then
the folds swap; the cross-fitted EPDMS is the average of the two held-out values. Paired bootstrap
(groups resampled with replacement, same draw for every arm) gives a 95% interval for every arm and
for its difference to the stock run.
Usage: crossfit_bootstrap.py <groups_dir> <stock_run> <candidate_run> [<candidate_run> ...] [--boot 10000]
  (run names without the groups_ prefix / .npz suffix)
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

argv = sys.argv[1:]
B = int(argv[argv.index("--boot") + 1]) if "--boot" in argv else 10000
argv = [a for i, a in enumerate(argv) if a != "--boot" and (i == 0 or argv[i - 1] != "--boot")]
gdir, stock, cands = Path(argv[0]), argv[1], argv[2:]


def load(run):
    d = np.load(gdir / f"groups_{run}.npz")
    return list(d["gid"]), d["score"].astype(float)


gid, s_stock = load(stock)
S = {c: load(c)[1] for c in cands}
for c in cands:
    assert len(S[c]) == len(s_stock), c
fold = np.array([int(hashlib.md5(g.encode()).hexdigest(), 16) % 2 for g in gid])
A, Bf = fold == 0, fold == 1
print(f"{len(gid)} groups, fold sizes {A.sum()} / {Bf.sum()}; stock {stock} = {s_stock.mean():.4f}")

# ---- cross-fitted selection ----
held = []
for sel, rep, tag in ((A, Bf, "A->B"), (Bf, A, "B->A")):
    best = max(cands, key=lambda c: S[c][sel].mean())
    held.append(S[best][rep].mean())
    print(f"  select on {tag[0]}: {best} ({S[best][sel].mean():.4f} on {tag[0]}) -> held-out {tag[-1]} {S[best][rep].mean():.4f}  "
          f"(stock on {tag[-1]}: {s_stock[rep].mean():.4f})")
full_best = max(cands, key=lambda c: S[c].mean())
print(f"cross-fitted EPDMS = {np.mean(held):.4f}   full-split best = {full_best} {S[full_best].mean():.4f}   "
      f"search space = {len(cands)} arms")

# ---- paired bootstrap ----
rng = np.random.default_rng(0)
idx = rng.integers(0, len(gid), size=(B, len(gid)))
bs = s_stock[idx].mean(1)
print(f"\n| arm | EPDMS | 95% CI | Δ vs stock | 95% CI of Δ | P(Δ<=0) |\n|---|---|---|---|---|---|")
print(f"| {stock} | {s_stock.mean():.4f} | [{np.percentile(bs,2.5):.4f}, {np.percentile(bs,97.5):.4f}] | | | |")
for c in cands:
    bc = S[c][idx].mean(1); d = bc - bs
    print(f"| {c} | {S[c].mean():.4f} | [{np.percentile(bc,2.5):.4f}, {np.percentile(bc,97.5):.4f}] | "
          f"{100*(S[c].mean()-s_stock.mean()):+.2f} | [{100*np.percentile(d,2.5):+.2f}, {100*np.percentile(d,97.5):+.2f}] | {(d<=0).mean():.3f} |")
