#!/usr/bin/env bash
# Figure 2: strength sweep of the learned drivable-area energy (margin 1.5), seed 0 / 42. Operating points are in table1.sh.
set -euo pipefail; source "$(dirname "$0")/common.sh"
for L in 1.00 2.00 4.00 6.00 8.00 12.00; do r drivewam dwam_cornerlearn_m1.5_l$L cornerlearn $L CORNER_MARGIN=1.5; done
for L in 0.01 0.02 0.05 0.10 0.20; do
  r simwam   simsup_cornerlearn_m1.5_l$L cornerlearn $L CORNER_MARGIN=1.5
  r drivelaw dlaw_cornerlearn_m1.5_l$L   cornerlearn $L CORNER_MARGIN=1.5
done
