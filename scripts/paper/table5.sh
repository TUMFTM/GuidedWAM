#!/usr/bin/env bash
# Table 5: Jacobian-free vs full-Jacobian guidance (learned field, margin 1.5) with per-step timing and peak memory.
# The Jacobian-free rows are the fig2.sh operating points rerun with GUIDANCE_STATS=1.
set -euo pipefail; source "$(dirname "$0")/common.sh"
for J in 0 1; do sfx=; [ $J = 1 ] && sfx=_jac
  r drivewam dwam_cornerlearn_m1.5_l4.00_stats$sfx    cornerlearn 4.00 CORNER_MARGIN=1.5 JAC=$J GUIDANCE_STATS=1
  r simwam   simsup_cornerlearn_m1.5_l0.05_stats$sfx  cornerlearn 0.05 CORNER_MARGIN=1.5 JAC=$J GUIDANCE_STATS=1
  r drivelaw dlaw_cornerlearn_m1.5_l0.05_stats$sfx    cornerlearn 0.05 CORNER_MARGIN=1.5 JAC=$J GUIDANCE_STATS=1
done
"$SIMWAM_VENV/bin/python" "$GUIDEDWAM_ROOT/scripts/ablation/guidance_stats_summary.py" "$GUIDEDWAM_EXP/navhard_validate"/*_stats*
