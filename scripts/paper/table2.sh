#!/usr/bin/env bash
# Table 2: guidance vs post-hoc gradient projection, both on the privileged drivable area (margin 0.5 m).
# Needs: table1.sh (stock runs), scripts/fields/build_fields.sh oracle.
set -euo pipefail; source "$(dirname "$0")/common.sh"
r drivewam dwam_corner_l4.00    corner 4.00 CORNER_MARGIN=0.5
r simwam   simsup_corner_l0.05  corner 0.05 CORNER_MARGIN=0.5
r drivelaw dlaw_corner_l0.05    corner 0.05 CORNER_MARGIN=0.5
for m in dwam simsup dlaw; do                       # projection of the stock plans (margin 0.5 fixed in the script)
  SRC=$GUIDEDWAM_EXP/navhard_validate/${m}_stock/pred_actions; OUT=$GUIDEDWAM_EXP/navhard_proj/${m}_stock_proj/pred_actions
  [ -d "$OUT" ] || "$FIELDS_VENV/bin/python" "$GUIDEDWAM_ROOT/scripts/ablation/project_navhard.py" "$SRC" "$OUT" "$FIELDS_DIR"
  bash "$NAVHARD/score.sh" proj_${m}_stock "$OUT"
done
