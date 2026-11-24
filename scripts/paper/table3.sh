#!/usr/bin/env bash
# Table 3: energy composition on privileged information. LK = map (lane centre), DA = corner (drivable area, m0.5),
# DD = offroute (driving direction). Composed arms weight the energies with W_DAC (first) / W_NC (second), CORNER_W (DA).
# Seeds as in the paper runs; the table reports their mean where more than one is listed.
set -euo pipefail; source "$(dirname "$0")/common.sh"
# DriveWAM
r drivewam dwam_map_l2.00                     map               2.00
r drivewam dwam_offroute_l0.24                offroute          0.24 SEED=0
r drivewam dwam_offroute_l0.24_seed1          offroute          0.24 SEED=1
r drivewam dwam_mapcorner_l2.00               mapcorner         2.00
r drivewam dwam_corneroffroute_l4.00_w0.06    corneroffroute    4.00 CORNER_MARGIN=0.5 W_NC=0.06 SEED=0
r drivewam dwam_corneroffroute_l4.00_w0.06_seed1 corneroffroute 4.00 CORNER_MARGIN=0.5 W_NC=0.06 SEED=1
r drivewam dwam_mapcorneroffroute_l2.00_w0.12 mapcorneroffroute 2.00 CORNER_W=1 W_NC=0.12 SEED=0
# SimWAM
r simwam simsup_map_l0.02                     map               0.02
r simwam simsup_offroute_l0.003               offroute          0.003
r simwam simsup_offroute_l0.003_seed1         offroute          0.003 SEED=1
r simwam simsup_mapcorner_l0.02               mapcorner         0.02
r simwam simsup_corneroffroute_l0.05_w0.06    corneroffroute    0.05 CORNER_MARGIN=0.5 W_NC=0.06
r simwam simsup_corneroffroute_l0.05_w0.06_seed1 corneroffroute 0.05 CORNER_MARGIN=0.5 W_NC=0.06 SEED=1
r simwam simsup_mapcorneroffroute_l0.02_w0.15 mapcorneroffroute 0.02 CORNER_W=1 W_NC=0.15
# DriveLaW
r drivelaw dlaw_map_l0.02                     map               0.02
for SD in 42 1 2; do r drivelaw dlaw_offroute_l0.0030_seed$SD offroute 0.003 SEED=$SD; done
r drivelaw dlaw_mapcorner_w2_l0.02            mapcorner         0.02 W_NC=2
for SD in 42 1 2; do r drivelaw dlaw_corneroffroute_l0.05_w0.06_seed$SD corneroffroute 0.05 W_NC=0.06 SEED=$SD; done
r drivelaw dlaw_mapcorneroffroute_w2_l0.02_w0.15 mapcorneroffroute 0.02 CORNER_W=2 W_NC=0.15
