# Source this before any script:  source scripts/env.sh
# Machine-specific locations go in scripts/local.env (not tracked); see docs/installation.md.
GUIDEDWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GUIDEDWAM_ROOT
[ -f "$GUIDEDWAM_ROOT/scripts/local.env" ] && source "$GUIDEDWAM_ROOT/scripts/local.env"

# NAVSIM dataset (maps/, navsim_logs/, sensor_blobs/, navhard_two_stage/)
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$GUIDEDWAM_ROOT/data/navsim}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NAVHARD_SYNTHETIC_SENSOR_PATH="${NAVHARD_SYNTHETIC_SENSOR_PATH:-$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs}"
export NAVHARD_SYNTHETIC_SCENES_PATH="${NAVHARD_SYNTHETIC_SCENES_PATH:-$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles}"
export NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-$OPENSCENE_DATA_ROOT/navsim_logs/test}"
export NAVSIM_SENSOR_BLOBS_PATH="${NAVSIM_SENSOR_BLOBS_PATH:-$OPENSCENE_DATA_ROOT/sensor_blobs/test}"

# weights (DriveWAM/, lingbot-va-base/, DriveLaW/, SimWAM/, simwam_checkpoints/, ltf_seed_0.ckpt)
export GUIDEDWAM_WEIGHTS="${GUIDEDWAM_WEIGHTS:-$GUIDEDWAM_ROOT/weights}"
export DRIVELAW_WEIGHTS="${DRIVELAW_WEIGHTS:-$GUIDEDWAM_WEIGHTS/DriveLaW}"

# experiment outputs and caches
export GUIDEDWAM_EXP="${GUIDEDWAM_EXP:-$GUIDEDWAM_ROOT/exp}"
export NAVHARD_CACHE="${NAVHARD_CACHE:-$GUIDEDWAM_EXP/metric_cache_navhard_two_stage}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$GUIDEDWAM_EXP/scores}"
export FIELDS_DIR="${FIELDS_DIR:-$GUIDEDWAM_EXP/navhard_fields}"
export LEARNED_FIELDS="${LEARNED_FIELDS:-$GUIDEDWAM_EXP/bevsdf_fields_all}"

# one virtualenv per model (envs/*.lock.txt); "fields" builds the oracle and learned fields
export DRIVEWAM_VENV="${DRIVEWAM_VENV:-$GUIDEDWAM_ROOT/.venvs/drivewam}"
export SIMWAM_VENV="${SIMWAM_VENV:-$GUIDEDWAM_ROOT/.venvs/simwam}"
export DRIVELAW_VENV="${DRIVELAW_VENV:-$GUIDEDWAM_ROOT/.venvs/drivelaw}"
export FIELDS_VENV="${FIELDS_VENV:-$GUIDEDWAM_ROOT/.venvs/fields}"

# navsim first (the scorer's devkit), then the guidedwam package
export PYTHONPATH="$GUIDEDWAM_ROOT/third_party/navsim:$GUIDEDWAM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DRIVEWAM_TEXT_ENCODER_DEVICE="${DRIVEWAM_TEXT_ENCODER_DEVICE:-cpu}"
export DRIVELAW_OFFLOAD_TEXT_ENCODER="${DRIVELAW_OFFLOAD_TEXT_ENCODER:-1}"
