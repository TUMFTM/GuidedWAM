#!/usr/bin/env bash
# Link the downloaded weights and the dataset into the places the baseline repos load them from.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
W=$GUIDEDWAM_WEIGHTS; T=$GUIDEDWAM_ROOT/third_party
lnk() { mkdir -p "$(dirname "$2")"; ln -sfn "$1" "$2"; echo "$2 -> $1"; }

# DriveWAM: ./weights/lingbot-va-base and ./weights/DriveWAM/navsim_ckpt/transformer, relative to the repo
lnk "$W/lingbot-va-base" "$T/drivewam/weights/lingbot-va-base"
lnk "$W/DriveWAM/navsim" "$T/drivewam/weights/DriveWAM/navsim_ckpt/transformer"
# SimWAM: ./weights/SimWAM.pt, ./checkpoints/ (Wan2.2 + ActionDiT), ./data/navsim
lnk "$W/SimWAM/weights/SimWAM.pt" "$T/simwam/weights/SimWAM.pt"
lnk "$W/simwam_checkpoints" "$T/simwam/checkpoints"
lnk "$OPENSCENE_DATA_ROOT" "$T/simwam/data/navsim"
# DriveLaW reads $DRIVELAW_WEIGHTS through configs/drivelaw_eval_navsim.yaml
