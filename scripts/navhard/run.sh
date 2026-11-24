#!/usr/bin/env bash
# Predict the full navhard split (5,912 scenes) with one WAM and one guidance arm, then score it.
#   MODEL=drivewam|simwam|drivelaw NAME=<run> [ARM=off LAM=0 W_DAC=1 W_NC=1 CORNER_MARGIN= CORNER_W= SEED=
#     JAC=0 WINDOW=asymmetric STEPS= GUIDANCE_STATS= TOKENS_FILE= NO_SCORE=0] scripts/navhard/run.sh
# Arms: off | corner (oracle drivable area) | cornerlearn (learned camera field) | map (lane keeping) |
#   offroute (driving direction) | mapcorner | corneroffroute | mapcorneroffroute ...
#   W_DAC / W_NC weight the first / second energy of a composed arm.
# Output: $GUIDEDWAM_EXP/navhard_validate/<NAME>/pred_actions/<token>.npy, (8, 3) poses at 0.5 s.
# SEED defaults: DriveWAM 0, SimWAM 0, DriveLaW 42 (the stock runs).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
S=$GUIDEDWAM_ROOT/scripts/navhard
MODEL=${MODEL:?drivewam|simwam|drivelaw}; NAME=${NAME:?}
ARM=${ARM:-off}; LAM=${LAM:-0}; W_DAC=${W_DAC:-1}; W_NC=${W_NC:-1}
PRED=$GUIDEDWAM_EXP/navhard_validate/$NAME/pred_actions; mkdir -p "$PRED"
[ -n "${CORNER_MARGIN:-}" ] && export CORNER_MARGIN
[ -n "${CORNER_W:-}" ] && export CORNER_W
[ -n "${GUIDANCE_STATS:-}" ] && export GUIDANCE_STATS
SUBSET=(ALL_TOKENS=1); [ -n "${TOKENS_FILE:-}" ] && SUBSET=(TOKENS_FILE="$TOKENS_FILE")
echo "===== $NAME predictions START $(date -u) ($MODEL arm=$ARM lam=$LAM w=$W_DAC/$W_NC margin=${CORNER_MARGIN:-default} seed=${SEED:-default}) ====="
case $MODEL in
drivewam)
  cd "$GUIDEDWAM_ROOT/third_party/drivewam"   # loads ./weights relative to the repo
  if [ "$ARM" = off ] && [ "${SEED:-0}" = 0 ]; then   # stock seed 0; other stock seeds go through the guided script
    env PRED_DIR="$PRED" ${TOKENS_FILE:+TOKENS_FILE="$TOKENS_FILE"} "$DRIVEWAM_VENV/bin/python" "$S/predict_navhard_drivewam.py"
  else
    env "${SUBSET[@]}" OUT_DIR="$PRED" GUIDANCE_ARM="$ARM" GUIDANCE_LAM="$LAM" FS_W_DAC="$W_DAC" FS_W_NC="$W_NC" \
      PRED_SEED="${SEED:-0}" JACOBIAN="${JAC:-0}" WINDOW="${WINDOW:-asymmetric}" NUM_ACTION_STEPS="${STEPS:-10}" \
      "$DRIVEWAM_VENV/bin/python" "$S/guided_navhard_drivewam.py"
  fi ;;
simwam)
  source "$SIMWAM_VENV/bin/activate"
  cd "$GUIDEDWAM_ROOT/third_party/simwam"
  ARGS=(EVALUATION.num_inference_steps="${STEPS:-20}")
  [ -n "${SEED:-}" ] && ARGS+=(EVALUATION.seed="$SEED")
  [ -n "${TOKENS_FILE:-}" ] && ARGS+=(EVALUATION.tokens_file="$TOKENS_FILE")
  if [ "$ARM" != off ]; then
    B() { [ "${1:-0}" = 1 ] && echo true || echo false; }
    ARGS+=(EVALUATION.guidance_arm="$ARM" EVALUATION.guidance_lam="$LAM" EVALUATION.guidance_w_dac="$W_DAC"
           EVALUATION.guidance_w_nc="$W_NC" EVALUATION.guidance_fields_dir="$FIELDS_DIR"
           +EVALUATION.guidance_window="${WINDOW:-asymmetric}" +EVALUATION.guidance_jacobian="$(B "${JAC:-0}")"
           +EVALUATION.guidance_normalize="$(B "${NORM:-0}")" +EVALUATION.bestofn=0 +EVALUATION.bestofn_seed0=0)
  fi
  NPROC_PER_NODE=1 CKPT=./weights/SimWAM.pt EXP_NAME="$NAME" PRED_DIR="$PRED" \
    bash experiments/navsim/run_predict_navhard.sh "${ARGS[@]}" ;;
drivelaw)
  cd "$GUIDEDWAM_EXP"   # any directory without a navsim/ that would shadow third_party/navsim
  env SPLIT=navhard "${SUBSET[@]}" OUT_DIR="$PRED" GUIDANCE_ARM="$ARM" GUIDANCE_LAM="$LAM" FS_W_DAC="$W_DAC" FS_W_NC="$W_NC" \
    EGO_HALF_W="${EGO_HALF_W:-1.0}" CORNER_MARGIN="${CORNER_MARGIN:-0.5}" DRIVELAW_SEED="${SEED:-42}" \
    DRIVELAW_JACOBIAN="${JAC:-0}" "$DRIVELAW_VENV/bin/python" "$S/predict_navhard_drivelaw.py" ;;
*) echo "unknown MODEL $MODEL" >&2; exit 2 ;;
esac
echo "  predictions written: $(ls "$PRED" | wc -l)"
[ "${NO_SCORE:-0}" = 1 ] || bash "$S/score.sh" "$NAME" "$PRED"
