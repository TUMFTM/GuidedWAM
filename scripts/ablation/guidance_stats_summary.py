"""Summarise guidance_stats.jsonl files (plan §2 instrumentation, §4 breaker/clip rates, §5 timing).

Per run: mean over scenes of the correction/velocity ratio at the window PEAK (steps where lam == max lam
of that scene), the mean per-step metric endpoint displacement, the clip-bind rate, and the forward vs
guidance time / peak memory per sampler step. Prints a markdown table; optional CSV.
Usage: guidance_stats_summary.py <dir-or-jsonl> [more ...] [--csv out.csv]
"""
import glob
import json
import os
import sys

import numpy as np

args = [a for a in sys.argv[1:] if a != "--csv"]
csv_out = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else None
if csv_out:
    args = [a for a in args if a != csv_out]
paths = []
for a in args:
    paths += [a] if a.endswith(".jsonl") else sorted(glob.glob(os.path.join(a, "**", "guidance_stats.jsonl"), recursive=True))

rows = []
for p in paths:
    name = os.path.basename(os.path.dirname(p))
    ratio_peak, ratio_all, d_end, d_mean, bind, n_steps = [], [], [], [], [], []
    t_fwd, t_guid, m_fwd, m_peak, jac = [], [], [], [], []
    n = 0
    for line in open(p):
        r = json.loads(line); st = r["steps"]
        if not st:
            continue
        n += 1
        lam = np.array([s["lam"] for s in st]); rt = np.array([s["ratio"] for s in st])
        on = lam > 0
        peak = np.isclose(lam, lam.max()) & on
        ratio_peak.append(rt[peak].mean() if peak.any() else 0.0)
        ratio_all.append(rt[on].mean() if on.any() else 0.0)
        d_end.append(np.sum([s["d_end"] for s in st])); d_mean.append(np.mean([s["d_mean"] for s in st if s["lam"] > 0] or [0]))
        bind.append(np.mean([s["bind"] for s in st if s["lam"] > 0] or [0])); n_steps.append(int(on.sum()))
        for t in r.get("timing", []):
            t_fwd.append(t["t_fwd"]); t_guid.append(t["t_guid"]); m_fwd.append(t["mem_fwd_gb"]); m_peak.append(t["mem_peak_gb"]); jac.append(t["jacobian"])
    if n == 0:
        continue
    rows.append({"run": name, "scenes": n, "guided_steps": float(np.mean(n_steps)),
                 "ratio_peak": float(np.mean(ratio_peak)), "ratio_peak_med": float(np.median(ratio_peak)),
                 "ratio_all": float(np.mean(ratio_all)), "endpoint_shift_sum_m": float(np.mean(d_end)),
                 "per_step_shift_m": float(np.mean(d_mean)), "clip_bind_rate": float(np.mean(bind)),
                 "t_fwd_ms": 1e3 * float(np.mean(t_fwd)) if t_fwd else np.nan,
                 "t_guid_ms": 1e3 * float(np.mean(t_guid)) if t_guid else np.nan,
                 "mem_fwd_gb": float(np.max(m_fwd)) if m_fwd else np.nan, "mem_peak_gb": float(np.max(m_peak)) if m_peak else np.nan,
                 "jacobian_steps": float(np.mean(jac)) if jac else np.nan})

cols = ["run", "scenes", "guided_steps", "ratio_peak", "ratio_peak_med", "ratio_all", "endpoint_shift_sum_m", "per_step_shift_m",
        "clip_bind_rate", "t_fwd_ms", "t_guid_ms", "mem_fwd_gb", "mem_peak_gb", "jacobian_steps"]
print("| " + " | ".join(cols) + " |"); print("|" + "---|" * len(cols))
for r in rows:
    print("| " + " | ".join(str(r[c]) if isinstance(r[c], (str, int)) else f"{r[c]:.4g}" for c in cols) + " |")
if csv_out:
    import pandas as pd
    pd.DataFrame(rows)[cols].to_csv(csv_out, index=False); print("wrote", csv_out)
