#!/usr/bin/env bash
# Two-stage EPDMS of one prediction directory on the full navhard split (CPU, simwam venv).
#   scripts/navhard/score.sh <name> [<pred_dir>]   (default pred_dir: $GUIDEDWAM_EXP/navhard_validate/<name>/pred_actions)
# The scorer writes $NAVSIM_EXP_ROOT/val_<name>_navhard_v2/<timestamp>/<timestamp>.csv.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
NAME=${1:?name}; PRED=${2:-$GUIDEDWAM_EXP/navhard_validate/$NAME/pred_actions}
echo "===== SCORE $NAME ($(ls "$PRED" | wc -l) predictions) START $(date -u) ====="
source "$SIMWAM_VENV/bin/activate"
SYNTHETIC_SENSOR_PATH=$NAVHARD_SYNTHETIC_SENSOR_PATH SYNTHETIC_SCENES_PATH=$NAVHARD_SYNTHETIC_SCENES_PATH \
SPLIT=navhard EXP_NAME="val_$NAME" NAVHARD_PRED_DIR="$PRED" \
bash "$GUIDEDWAM_ROOT/third_party/navsim/scripts/evaluation/run_npy_trajectory_agent_pdm_score_evaluation.sh" 2>&1 \
  | grep -E "Final extended|Error|Traceback" | tail -2
echo "##### SCORE $NAME DONE $(date -u) #####"
