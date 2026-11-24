#!/usr/bin/env bash
# Precompute every field the energies read, for all 5,912 navhard scenes.
#   scripts/fields/build_fields.sh oracle    # privileged map fields: ora_/lanes_/route_<token>.npz in $FIELDS_DIR
#   scripts/fields/build_fields.sh learned   # camera-only BEV signed-distance field: train, then bsdf_<token>.npz in $LEARNED_FIELDS
# The learned field warm-starts from NAVSIM's LTF checkpoint ($GUIDEDWAM_WEIGHTS/ltf_seed_0.ckpt) and trains on navtrain
# (needs the navtrain logs and trainval sensor blobs). Its checkpoint lands in $GUIDEDWAM_EXP/bevsdf_run1/best.pt.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
F=$GUIDEDWAM_ROOT/scripts/fields; TOK=$GUIDEDWAM_ROOT/data/scene_lists
mkdir -p "$FIELDS_DIR" "$GUIDEDWAM_EXP"; cd "$GUIDEDWAM_EXP"
shards() { # N script [env...]: run N shards in parallel
  local n=$1 py=$2 script=$3; shift 3; local pids=()
  for ((s=0; s<n; s++)); do env "$@" SHARD=$s NSHARD=$n "$py" "$script" > "$(basename "$script" .py)_s$s.log" 2>&1 & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p"; done; }

case ${1:?oracle|learned} in
oracle)
  ALL_TOKENS=1 OUT_DIR="$FIELDS_DIR" "$DRIVEWAM_VENV/bin/python" "$F/precompute_oracle_fields.py" 2>&1 | tail -2   # ora_ (drivable area)
  ALL_TOKENS=1 LANE_OUT="$FIELDS_DIR" "$FIELDS_VENV/bin/python" "$F/precompute_lane_fields.py" 2>&1 | tail -2      # lanes_ (lane centrelines)
  shards 6 "$FIELDS_VENV/bin/python" "$F/precompute_route_fields.py" OUT_DIR="$FIELDS_DIR"                          # route_ (driving direction)
  echo "oracle fields: $(ls "$FIELDS_DIR" | wc -l) files" ;;
learned)
  for SPLIT in navtrain navhard; do shards 8 "$FIELDS_VENV/bin/python" "$F/bevsdf_targets.py" SPLIT=$SPLIT; done   # SDF targets from the map
  SPLIT=navtrain "$FIELDS_VENV/bin/python" "$F/bevsdf_index.py"                                                     # bevsdf_index_navtrain.npz
  shards 16 "$FIELDS_VENV/bin/python" "$F/bevsdf_cache_images.py"                                                   # bevsdf_images.npy
  "$FIELDS_VENV/bin/python" "$F/bevsdf_train.py"                                                                    # 20 epochs -> bevsdf_run1/
  TOKENS_FILE=$TOK/navhard_all_tokens.json SAVE_FIELDS="$LEARNED_FIELDS" OUT="$GUIDEDWAM_EXP/bevsdf_eval_alltok.json" \
    "$FIELDS_VENV/bin/python" "$F/bevsdf_eval.py"                                                                   # bsdf_<token>.npz
  echo "learned fields: $(ls "$LEARNED_FIELDS" | wc -l) files" ;;
esac
