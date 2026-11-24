"""Full per-metric NAVSIM v2 results for EVERY scored navhard run -> RESULTS_PER_METRIC.md + .csv.
Reads the latest scoring CSV of each run dir in evaluate_results_v2/ (same source as gen_results_md.py).
One table per (model, stage) with all nine sub-metrics and the stage EPDMS; stage-two tables also carry
the combined EPDMS. Run after every new full-split run."""
import glob
import os
from datetime import date

import pandas as pd
from guidedwam import paths

EV = str(paths.SCORES) + "/"
OUT_MD = str(paths.EXP / "RESULTS_PER_METRIC.md")
OUT_CSV = str(paths.EXP / "RESULTS_PER_METRIC.csv")
G = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance"]
W = ["ego_progress", "time_to_collision_within_bound", "lane_keeping", "history_comfort", "two_frame_extended_comfort"]
SHORT = {"no_at_fault_collisions": "NC", "drivable_area_compliance": "DAC", "driving_direction_compliance": "DDC",
         "traffic_light_compliance": "TLC", "ego_progress": "EP", "time_to_collision_within_bound": "TTC",
         "lane_keeping": "LK", "history_comfort": "HC", "two_frame_extended_comfort": "EC"}
STAGES = ["stage_one", "stage_two"]


def model_of(run):
    r = run.replace("val_val_", "").replace("val_", "")
    if r.startswith("proj_"):
        return "post-hoc projection (all models)"
    if "dlaw" in r or "drivelaw" in r:
        return "DriveLaW"
    if "simsup" in r or (r.startswith("simwam") and "rl" not in r and "fs_guided" not in r and "fsseg" not in r):
        return "SimWAM.pt (supervised)"
    if "simwam" in r:
        return "SimWAM-RL"
    if "dwam" in r or "drivewam" in r:
        return "DriveWAM"
    return "other"


def load(run, root):
    fs = sorted(glob.glob(root + run + "/*/*.csv"))
    if not fs:
        return None
    df = pd.read_csv(fs[-1])
    agg = df[df.token.astype(str).str.startswith("extended")].set_index("token")
    n = int((~df.token.astype(str).str.startswith("extended")).sum())
    return agg, n, os.path.basename(fs[-1])


# DriveLaW's validate script scores into its own root (drivelaw_env.sh); same run name in both -> newest CSV wins
ROOTS = [EV, str(paths.EXP / "drivelaw_out/")]
found = {}
for root in ROOTS:
    for run in sorted(os.listdir(root)):
        r = load(run, root)
        if r is not None and (run not in found or r[2] > found[run][2]):
            found[run] = r
rows = []
for run in sorted(found):
    agg, n, csv = found[run]
    rec = {"model": model_of(run), "run": run, "scenes": n, "csv": csv,
           "EPDMS": agg.loc["extended_pdm_score_combined", "score"]}
    for st in STAGES:
        rec[f"EPDMS_{st}"] = agg.loc[f"extended_pdm_score_{st}", "score"]
        for m in G + W:
            rec[f"{SHORT[m]}_{st}"] = agg.loc[f"extended_pdm_score_{st}", f"{m}_{st}"]
    rows.append(rec)
df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False, float_format="%.4f")

ORDER = ["DriveWAM", "SimWAM.pt (supervised)", "DriveLaW", "post-hoc projection (all models)", "SimWAM-RL", "other"]
md = [f"# Full per-metric NAVSIM v2 results — navhard_two_stage (5,912 scenes)\n",
      f"_Generated {date.today()} by `gen_per_metric_md.py` from `evaluate_results_v2/<run>/<ts>/<ts>.csv` "
      f"({len(df)} runs). Machine-readable copy: `RESULTS_PER_METRIC.csv`._\n",
      "EPDMS = NC·DAC·DDC·TLC × weighted(EP 5, TTC 5, LK 2, HC 2, EC 2). Stage one = 450 real-image scenes; "
      "stage two = 5,462 3DGS-rendered scenes; combined = the two-stage pseudo-closed-loop aggregate the "
      "leaderboard reports. Every value is the split-level aggregate the scorer prints, not a per-scene mean. "
      "Run names: `dwam` DriveWAM, `simsup` SimWAM.pt, `dlaw` DriveLaW, `proj_` post-hoc projection; `cornerlearn` = "
      "learned camera-only field, `corner` = oracle map field, `offroute` = driving-direction energy, "
      "`m<x>` margin, `l<x>` λ, `w<x>` second-energy weight, `seed<k>`/`_s<k>` replicate.\n"]
for model in ORDER:
    sub = df[df.model == model]
    if sub.empty:
        continue
    md.append(f"\n## {model} ({len(sub)} runs)\n")
    for st in STAGES:
        cols = [f"{SHORT[m]}_{st}" for m in G + W]
        hdr = "| run | " + " | ".join(SHORT[m] for m in G + W) + f" | **EPDMS {st.replace('_', ' ')}** |"
        sep = "|---|" + "---|" * (len(cols) + 1)
        if st == "stage_two":
            hdr += " **EPDMS combined** |"; sep += "---|"
        md.append(f"\n### {model} — {st.replace('_', ' ')}\n")
        md.append(hdr); md.append(sep)
        for _, r in sub.iterrows():
            line = f"| `{r.run}` | " + " | ".join(f"{r[c]:.4f}" for c in cols) + f" | **{r[f'EPDMS_{st}']:.4f}** |"
            if st == "stage_two":
                line += f" **{r.EPDMS:.4f}** |"
            md.append(line)
with open(OUT_MD, "w") as f:
    f.write("\n".join(md) + "\n")
print(f"{len(df)} runs -> {OUT_MD}, {OUT_CSV}")
print(df.groupby("model").size())
