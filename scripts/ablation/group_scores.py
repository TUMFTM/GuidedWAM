"""Per-group EPDMS for a scored run (the 225 pseudo-closed-loop groups navsim_v2 averages), for the
cross-fitted selection and the paired bootstrap of the ablation plan (§2).

Re-implements the two-stage aggregation exactly as cf_two_frame.py (verified to 1e-5 against the
scorer) and writes <out_dir>/groups_<run>.npz with: group ids, per-group score, the run's combined
EPDMS from the CSV, and the reconstructed value.
Usage: group_scores.py <run_dir> <pred_dir> <out_dir>
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

import navsim
assert "third_party/navsim/" in str(navsim.__file__)
from navsim.common.dataloader import MetricCacheLoader
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from guidedwam import paths

CACHE = str(paths.METRIC_CACHE)
MAPPING = str(paths.NAVSIM / "navsim/planning/script/config/common/train_test_split/navhard_two_stage.yaml")
run_dir, pred_dir, out_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)
GATES = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance"]
WM = ["ego_progress", "time_to_collision_within_bound", "lane_keeping", "history_comfort", "two_frame_extended_comfort"]
W = PDMScorer(proposal_sampling=TrajectorySampling(num_poses=40, interval_length=0.1))._config.weighted_metrics_array

df = pd.read_csv(sorted(glob.glob(run_dir + "/*/*.csv"))[-1])
agg = df[df.token.astype(str).str.startswith("extended")].set_index("token")
rows = df[~df.token.astype(str).str.startswith("extended")].set_index("token")
s1 = rows[GATES[0] + "_stage_one"].notna()
T = pd.DataFrame(index=rows.index); T["stage"] = np.where(s1, 1, 2)
for c in GATES + WM:
    T[c] = rows[c + "_stage_one"].where(s1, rows[c + "_stage_two"])
T["score"] = rows["score"]
rec = T[GATES].prod(axis=1).to_numpy() * (T[WM].to_numpy() * W[None]).sum(1) / W.sum()
assert np.abs(rec - T.score.to_numpy()).max() < 1e-4, "per-token reconstruction failed"

loader = MetricCacheLoader(Path(CACHE))
raw = OmegaConf.to_container(OmegaConf.load(MAPPING))["reactive_all_mapping"]
mapping = {(o, p): [tuple(x) for x in pairs] for o, p, pairs in raw if o in T.index or p in T.index}
start_pt, end_pt = {}, {}
for tok in T.index:
    ra = loader.get_from_token(tok).ego_state.rear_axle
    start_pt[tok] = (ra.x, ra.y)
    if T.loc[tok, "stage"] == 1:
        pl = np.load(pred_dir / f"{tok}.npy")
        ep = relative_to_absolute_poses(ra, [StateSE2(float(pl[-1, 0]), float(pl[-1, 1]), float(pl[-1, 2]))])[0]
        end_pt[tok] = (ep.x, ep.y)


def gw(s1_tok, s2_toks, sigma2=0.1):
    ex, ey = end_pt[s1_tok]
    d2 = np.array([(ex - start_pt[t][0]) ** 2 + (ey - start_pt[t][1]) ** 2 for t in s2_toks])
    w = np.exp(-d2 / (2 * sigma2))
    if np.isclose(w.sum(), 0.0) or np.isnan(w.sum()):
        return np.full(len(w), 1 / len(w))
    return w / w.sum()


s = pd.Series(rec, index=T.index)
gid, gsc = [], []
for (o, p), pairs in mapping.items():
    gs = []
    for s1_tok, s2_toks in [(o, [x[0] for x in pairs]), (p, [x[1] for x in pairs])]:
        s2_toks = [t for t in s2_toks if t in s.index]
        gs.append(s[s1_tok] * float((s[s2_toks].to_numpy() * gw(s1_tok, s2_toks)).sum()))
    gid.append(f"{o}|{p}"); gsc.append(np.mean(gs))
gsc = np.array(gsc); csv_c = float(agg.loc["extended_pdm_score_combined", "score"])
name = Path(run_dir.rstrip("/")).name
np.savez(out_dir / f"groups_{name}.npz", gid=np.array(gid), score=gsc, csv_combined=csv_c, recon=gsc.mean())
print(f"{name}: {len(gsc)} groups, recon {gsc.mean():.4f} vs csv {csv_c:.4f} (diff {gsc.mean()-csv_c:+.5f})")
