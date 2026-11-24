"""Locations of the baselines, data, weights and experiment outputs.

Everything outside the repository comes from the environment (scripts/env.sh sets all of it), so the
scripts carry no machine-specific paths.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO / "third_party"
DRIVEWAM = THIRD_PARTY / "drivewam"
DRIVELAW = THIRD_PARTY / "drivelaw"
SIMWAM = THIRD_PARTY / "simwam"
NAVSIM = THIRD_PARTY / "navsim"

# NAVSIM dataset root: maps/, navsim_logs/, sensor_blobs/, navhard_two_stage/
NAVSIM_DATA = Path(os.environ.get("OPENSCENE_DATA_ROOT", REPO / "data/navsim"))
# weights: DriveWAM/, DriveLaW/, SimWAM/, simwam_checkpoints/, ltf_seed_0.ckpt
WEIGHTS = Path(os.environ.get("GUIDEDWAM_WEIGHTS", REPO / "weights"))
# experiment outputs and caches: fields, predictions, scores
EXP = Path(os.environ.get("GUIDEDWAM_EXP", REPO / "exp"))
METRIC_CACHE = Path(os.environ.get("NAVHARD_CACHE", EXP / "metric_cache_navhard_two_stage"))
SCORES = Path(os.environ.get("NAVSIM_EXP_ROOT", EXP / "scores"))
