#!/usr/bin/env bash
# Figure 3: window shape (uniform vs trapezoid) against strength on 40 DriveWAM drivable-area failure scenes
# (oracle corner energy, margin 0.5): x0 traces per step, then the failure-rate and ADE/FDE plot.
set -euo pipefail; source "$(dirname "$0")/common.sh"
cd "$GUIDEDWAM_ROOT/third_party/drivewam"
TOKENS_FILE=$GUIDEDWAM_ROOT/data/scene_lists/x0_batch_tokens.json OUT_DIR=$GUIDEDWAM_EXP/x0_batch LAMS=4,8,16,32,64,128 \
  "$DRIVEWAM_VENV/bin/python" "$GUIDEDWAM_ROOT/scripts/ablation/x0_trace_batch.py"
cd "$GUIDEDWAM_EXP" && "$SIMWAM_VENV/bin/python" "$GUIDEDWAM_ROOT/scripts/figures/fig_window_shape.py" "$GUIDEDWAM_EXP/fig3_window_shape.pdf"
