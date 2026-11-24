# Sourced by the per-table scripts.  r MODEL NAME ARM LAM [VAR=value ...]  -> scripts/navhard/run.sh
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
NAVHARD=$GUIDEDWAM_ROOT/scripts/navhard
r() { local m=$1 n=$2 a=$3 l=$4; shift 4
  if [ -n "$(ls "$NAVSIM_EXP_ROOT"/val_${n}_navhard_v2/*/*.csv 2>/dev/null)" ] && [ "${FORCE:-0}" != 1 ]; then echo "skip $n (scored)"; return; fi
  env "$@" MODEL=$m NAME=$n ARM=$a LAM=$l bash "$NAVHARD/run.sh"; }
