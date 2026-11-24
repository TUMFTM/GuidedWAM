# Running the experiments

Every script sources `scripts/env.sh`, so it runs from any directory. A run predicts the full navhard split, all
5,912 scenes. The predictions land in `$GUIDEDWAM_EXP/navhard_validate/<name>/pred_actions/<token>.npy`, as
(8, 3) poses at 0.5 s in the ego frame. The run is then scored with the two-stage EPDMS into
`$NAVSIM_EXP_ROOT/val_<name>_navhard_v2/`. On one RTX 4090, prediction takes about 1.2 h per run for DriveWAM,
0.7 h for SimWAM and 1.7 h for DriveLaW. Scoring takes about 0.8 h on the CPU. The energies, the guided sampler and
the camera-only road-edge field are specified in [energies.md](energies.md).

## 0. Fields

The energies read precomputed fields, one file per scene.

```bash
scripts/fields/build_fields.sh oracle    # privileged map fields, Tables 2-3 and Figures 1/3: ora_/lanes_/route_<token>.npz
scripts/fields/build_fields.sh learned   # camera-only BEV signed-distance field, Tables 1/4/5 and Figure 2: bsdf_<token>.npz
```

The learned field is TransFuser's latent variant (LTF), warm-started from `ltf_seed_0.ckpt`. A head regresses the
signed distance to the road edge from the front cameras. It trains for 20 epochs on navtrain with targets drawn from
the map, and the checkpoint is `$GUIDEDWAM_EXP/bevsdf_run1/best.pt`. At inference it sees no map.

## 1. Paper tables and figures

| item | script | runs |
|---|---|---|
| Table 1: stock vs EG, learned drivable-area energy, 3 seeds | `scripts/paper/table1.sh` | 3 stock + 9 guided |
| Table 2: guidance vs post-hoc projection, oracle field, margin 0.5 m | `scripts/paper/table2.sh` | 3 guided + 3 projections |
| Table 3: composition of LK / DA / DD energies, privileged | `scripts/paper/table3.sh` | 24 |
| Figure 2: guidance strength | `scripts/paper/fig2.sh` | 16 |
| Table 4: cross-fitted strength search + paired bootstrap | `scripts/paper/table4.sh` | 6 extra + group scores |
| Figure 3: window shape vs strength, 40 DriveWAM DAC failures | `scripts/paper/fig3.sh` | per-step x0 traces |
| Table 5: Jacobian-free vs Jacobian, timing and memory | `scripts/paper/table5.sh` | 6 |
| Figure 1: front camera + BEV, stock / guided / projected | `scripts/paper/fig1.sh <token>` | rendering only |

The scripts skip a run that already has a score; `FORCE=1` reruns it. Table 2 needs Table 1's stock runs.
Tables 4 and 5 reuse runs from Table 1 and Figure 2.

### Operating points

| model | λ (learned field, Table 1) | corner margin | denoising steps | sampler seeds |
|---|---|---|---|---|
| DriveWAM | 4 | 1.0 m | 10 action steps | 0, 1, 2 |
| SimWAM (`SimWAM.pt`) | 0.05 | 1.5 m | 20 | 0, 1, 2 |
| DriveLaW | 0.05 | 1.5 m | 10 | 42, 1, 2 |

All runs use the trapezoidal window (`asymmetric`: zero above σ 0.95 and below 0.05, full from 0.8 to 0.3) and
the velocity bound c = 12. The privileged arms of Tables 2-3 use the oracle corner energy with a 0.5 m margin.

## 2. One run

```bash
MODEL=simwam NAME=my_run ARM=cornerlearn LAM=0.05 CORNER_MARGIN=1.5 SEED=0 scripts/navhard/run.sh
```

| variable | meaning |
|---|---|
| `MODEL` | `drivewam`, `simwam` or `drivelaw` |
| `ARM` | `off` (stock), `cornerlearn` (learned DA), `corner` (oracle DA), `map` (LK), `offroute` (DD), and the compositions `mapcorner`, `corneroffroute`, `mapcorneroffroute` |
| `LAM` | guidance strength λ |
| `W_DAC`, `W_NC` | weights of the first and second energy of a composed arm |
| `CORNER_MARGIN`, `CORNER_W` | drivable-area band and its weight in the three-term arm |
| `SEED` | sampler seed |
| `WINDOW` | `asymmetric` (trapezoid, default), `uniform`, `canonical`, `early`, `late` |
| `JAC=1` | full Jacobian (backprop through the denoiser) instead of the Jacobian-free form |
| `GUIDANCE_STATS=1` | per-step correction norms, timing and memory to `guidance_stats.jsonl` |
| `STEPS` | number of sampler steps (default: DriveWAM 10 action steps, SimWAM 20) |
| `SPEED_CAP`, `SPEED_CAP_W` | adds the speed-cap energy to any arm: cap in m/s and its weight. `ARM=speedcap` runs it alone |
| `SPEED_CAP_PRECOND=1` | displacement-space preconditioning of the speed-cap gradient, needed for SimWAM and DriveLaW |
| `NORM=1` | per-energy gradient normalization instead of fixed weights (stabilization ablation) |
| `GUIDANCE_CLIP_REL` | ratio c of the velocity bound (default 12) |
| `FIELD_SHIFT`, `FIELD_NOISE` | perturb the learned road-edge field by a shift or by smoothed noise, in m (sensitivity study) |
| `PROJ_SMOOTH` | weight of the smoothness term of the projection baseline (`scripts/ablation/project_navhard.py`) |
| `TOKENS_FILE` | json `[[token, ...], ...]`, restricts the run to those scenes (search only; report on the full split) |
| `NO_SCORE=1` | predict without scoring |

Score an existing prediction directory with `scripts/navhard/score.sh <name> <pred_dir>`. To collect every scored
run into one per-metric table, run `scripts/tables/gen_per_metric_md.py` with the simwam venv, which writes
`$GUIDEDWAM_EXP/RESULTS_PER_METRIC.{md,csv}`.

## Layout

| path | content |
|---|---|
| `guidedwam/energies/` | differentiable energies on the metric ego trajectory: map/camera fields (`navhard.py`, `navhard_fp.py`), kinematics (`kinematic.py`) |
| `guidedwam/sampler/` | the guided step and noise-level window (`guidance.py`), and per model the latent-to-metric adapter (`drivewam.py`, `simwam.py`, `drivelaw.py`); each baseline's `guidance.py` re-exports its adapter |
| `scripts/navhard/` | per-model prediction, `run.sh`, `score.sh` |
| `scripts/fields/` | oracle fields and the learned field |
| `scripts/ablation/` | projection, group scores, cross-fit, x0 traces, guidance statistics |
| `scripts/figures/`, `scripts/tables/` | Figure 1, Figure 3, per-metric tables |
| `scripts/paper/` | one script per table and figure |
| `configs/` | DriveLaW inference config |
| `data/scene_lists/` | lists of NAVSIM scene tokens (the scene identifiers of the dataset): all navhard scenes, the 40 Figure 3 scenes, Figure 1 candidates |
| `envs/` | pip lock files of the four environments |
