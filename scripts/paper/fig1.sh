#!/usr/bin/env bash
# Figure 1: front camera + BEV drivable-area field with the stock, guided and projected DriveWAM plans for one scene.
#   scripts/paper/fig1.sh <token>   (candidates: data/scene_lists/pov_bev_cases.json, case1_both_improve)
# Needs: table1.sh (dwam_stock), table2.sh (dwam_corner_l4.00, dwam_stock_proj).
set -euo pipefail; source "$(dirname "$0")/common.sh"
TOK=${1:?token}; V=$GUIDEDWAM_EXP/navhard_validate
STOCK_DIR=$V/dwam_stock/pred_actions GUIDED_DIR=$V/dwam_corner_l4.00/pred_actions \
PROJ_DIR=$GUIDEDWAM_EXP/navhard_proj/dwam_stock_proj/pred_actions CORNER_MARGIN=0.5 \
  "$FIELDS_VENV/bin/python" "$GUIDEDWAM_ROOT/scripts/figures/render_pov_bev_pair.py" "$TOK" "$GUIDEDWAM_EXP/fig1_$TOK.png"
