#!/usr/bin/env bash
# Table 4: cross-fitted strength/margin selection over the 225 navhard groups (two folds by group-id hash) with a
# 10,000-draw paired bootstrap. Needs: table1.sh and fig2.sh, plus the extra margin arms below.
set -euo pipefail; source "$(dirname "$0")/common.sh"
r drivewam dwam_cornerlearn_m0.5_l4.00 cornerlearn 4.00 CORNER_MARGIN=0.5
r drivewam dwam_cornerlearn_m2.0_l4.00 cornerlearn 4.00 CORNER_MARGIN=2.0
for L in 2.00 8.00; do r drivewam dwam_cornerlearn_m1.0_l$L cornerlearn $L CORNER_MARGIN=1.0; done
r simwam   simsup_cornerlearn_m1.0_l0.05 cornerlearn 0.05 CORNER_MARGIN=1.0
for M in 0.5 1.0; do r drivelaw dlaw_cornerlearn_m${M}_l0.05 cornerlearn 0.05 CORNER_MARGIN=$M; done

GS=$GUIDEDWAM_EXP/group_scores; mkdir -p "$GS"; A=$GUIDEDWAM_ROOT/scripts/ablation
group() { local n=$1; [ -f "$GS/groups_val_${n}_navhard_v2.npz" ] || "$SIMWAM_VENV/bin/python" "$A/group_scores.py" \
  "$NAVSIM_EXP_ROOT/val_${n}_navhard_v2" "$GUIDEDWAM_EXP/navhard_validate/$n/pred_actions" "$GS" | tail -1; }
cf() { local stock=$1; shift; group "$stock"; for c in "$@"; do group "$c"; done
  "$SIMWAM_VENV/bin/python" "$A/crossfit_bootstrap.py" "$GS" "val_${stock}_navhard_v2" $(printf "val_%s_navhard_v2 " "$@") --boot 10000; }
cf dwam_stock dwam_cornerlearn_m0.5_l4.00 dwam_cornerlearn_m1.0_l2.00 dwam_cornerlearn_m1.0_l4.00_seed0 dwam_cornerlearn_m1.0_l8.00 \
  dwam_cornerlearn_m1.5_l1.00 dwam_cornerlearn_m1.5_l2.00 dwam_cornerlearn_m1.5_l4.00 dwam_cornerlearn_m1.5_l6.00 \
  dwam_cornerlearn_m1.5_l8.00 dwam_cornerlearn_m1.5_l12.00 dwam_cornerlearn_m2.0_l4.00
cf simsup_stock simsup_cornerlearn_m1.0_l0.05 simsup_cornerlearn_m1.5_l0.01 simsup_cornerlearn_m1.5_l0.02 \
  simsup_cornerlearn_m1.5_l0.05 simsup_cornerlearn_m1.5_l0.10 simsup_cornerlearn_m1.5_l0.20
cf dlaw_stock dlaw_cornerlearn_m0.5_l0.05 dlaw_cornerlearn_m1.0_l0.05 dlaw_cornerlearn_m1.5_l0.01 dlaw_cornerlearn_m1.5_l0.02 \
  dlaw_cornerlearn_m1.5_l0.05 dlaw_cornerlearn_m1.5_l0.10 dlaw_cornerlearn_m1.5_l0.20
