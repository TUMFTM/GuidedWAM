#!/usr/bin/env bash
# Table 1: stock vs energy guidance with the camera-only learned drivable-area field (cornerlearn), three sampler seeds.
# Needs: scripts/fields/build_fields.sh learned.
set -euo pipefail; source "$(dirname "$0")/common.sh"
r drivewam dwam_stock      off 0
r simwam   simsup_stock    off 0
r drivelaw dlaw_stock      off 0
for SD in 0 1 2; do
  r drivewam dwam_cornerlearn_m1.0_l4.00_seed$SD  cornerlearn 4.00 CORNER_MARGIN=1.0 SEED=$SD
  r simwam   simsup_cornerlearn_m1.5_l0.05_seed$SD cornerlearn 0.05 CORNER_MARGIN=1.5 SEED=$SD
done
for SD in 42 1 2; do
  r drivelaw dlaw_cornerlearn_m1.5_l0.05_seed$SD  cornerlearn 0.05 CORNER_MARGIN=1.5 SEED=$SD
done
