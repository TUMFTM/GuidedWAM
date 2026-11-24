# Installation

GuidedWAM runs three frozen world action models (DriveWAM, SimWAM, DriveLaW) on the NAVSIM v2 navhard split. Each
model keeps its own Python environment, and a fourth environment builds the oracle and learned fields. Every
environment uses Python 3.10 and a CUDA GPU. The paper runs used one RTX 4090 (24 GB).

## 1. Clone

```bash
git clone https://github.com/TUMFTM/GuidedWAM.git
cd GuidedWAM
```

`third_party/` holds copies of DriveWAM, SimWAM and the planner part of DriveLaW, each with the sampling-time guidance
hook added, plus `navsim`, the NAVSIM v2 devkit that scores every run. Their sources, licenses and the changes made
here are listed in `third_party/README.md`.

## 2. Environments

The exact package versions of the paper runs are in `envs/<name>.lock.txt`. Create one virtualenv for each:

```bash
for e in drivewam simwam drivelaw fields; do
  python3.10 -m venv .venvs/$e
  .venvs/$e/bin/pip install --upgrade pip
done
.venvs/drivewam/bin/pip install -r envs/drivewam.lock.txt --extra-index-url https://download.pytorch.org/whl/cu126
.venvs/simwam/bin/pip   install -r envs/simwam.lock.txt   --extra-index-url https://download.pytorch.org/whl/cu126
.venvs/drivelaw/bin/pip install -r envs/drivelaw.lock.txt
.venvs/fields/bin/pip   install -r envs/fields.lock.txt
```

| env | used for | key versions |
|---|---|---|
| `drivewam` | DriveWAM predictions, oracle drivable-area fields, Figure 3 traces | torch 2.9.0+cu126, nuplan-devkit (git ce3c323) |
| `simwam` | SimWAM predictions, **all scoring**, group scores, tables | torch 2.7.1, nuplan-devkit 1.1.1 |
| `drivelaw` | DriveLaW predictions | torch 2.12.1, diffusers 0.40.0 |
| `fields` | lane and route fields, learned field (targets, training, inference), projection, Figure 1 | torch 2.0.1, nuplan-devkit 1.2.0 |

The baselines and `navsim` are not pip-installed. `scripts/env.sh` puts `third_party/navsim` and this repository on
`PYTHONPATH`, and each model script adds its own repository. The commented `-e` lines in the lock files are the
editable installs this replaces.

If your locations differ from the defaults below, set them in `scripts/local.env` (untracked). `scripts/env.sh`
reads that file, and every script sources `scripts/env.sh`:

```bash
OPENSCENE_DATA_ROOT=/data/navsim            # default ./data/navsim
GUIDEDWAM_WEIGHTS=/data/weights             # default ./weights
GUIDEDWAM_EXP=/data/guidedwam_exp           # default ./exp: fields, predictions, scores
DRIVEWAM_VENV=... SIMWAM_VENV=... DRIVELAW_VENV=... FIELDS_VENV=...   # default ./.venvs/<name>
```

## 3. NAVSIM data

Follow the [NAVSIM installation guide](https://github.com/autonomousvision/navsim/blob/main/docs/install.md) and
its download scripts (`download/` in that repository):

| download | needed for | lands in `$OPENSCENE_DATA_ROOT/` |
|---|---|---|
| `download_maps` | everything | `maps/` |
| `download_test` | navhard stage one (real scenes) | `navsim_logs/test`, `sensor_blobs/test` |
| `download_navhard_two_stage` | navhard stage two (synthetic scenes) | `navhard_two_stage/{sensor_blobs,synthetic_scene_pickles,openscene_meta_datas}` |
| `download_navtrain` | training the learned field only | `navsim_logs/trainval`, `sensor_blobs/trainval` |

Build the navhard metric cache once (CPU, about 3.5 h):

```bash
source scripts/env.sh && source $DRIVEWAM_VENV/bin/activate
cd third_party/navsim && python navsim/planning/script/run_metric_caching.py \
  train_test_split=navhard_two_stage metric_cache_path=$NAVHARD_CACHE \
  synthetic_sensor_path=$NAVHARD_SYNTHETIC_SENSOR_PATH synthetic_scenes_path=$NAVHARD_SYNTHETIC_SCENES_PATH
```

## 4. Weights

All weights are public. Put them under `$GUIDEDWAM_WEIGHTS` with this layout:

| model | source | files | path under `$GUIDEDWAM_WEIGHTS` |
|---|---|---|---|
| DriveWAM | [chenchenshi/DriveWAM](https://huggingface.co/chenchenshi/DriveWAM) | `navsim/*` | `DriveWAM/navsim/` |
| DriveWAM base | [robbyant/lingbot-va-base](https://huggingface.co/robbyant/lingbot-va-base) | `vae/*`, `tokenizer/*`, `text_encoder/*`, `*.json` | `lingbot-va-base/` |
| SimWAM | [H-EmbodVis/SimWAM](https://huggingface.co/H-EmbodVis/SimWAM) | `weights/SimWAM.pt` | `SimWAM/weights/SimWAM.pt` |
| SimWAM base | [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) through SimWAM's `scripts/model_prepare.sh` | video DiT, Wan2.2 VAE, umt5-xxl encoder, ActionDiT backbone | `simwam_checkpoints/` |
| DriveLaW | [tz2026/DriveLaW](https://huggingface.co/tz2026/DriveLaW) | `DriveLaW-Act/*`, `DriveLaW-Video/*` except `DriveLaW-Video/transformer/` | `DriveLaW/` |
| LTF (learned-field init) | [autonomousvision/navsim_baselines](https://huggingface.co/autonomousvision/navsim_baselines/tree/main/ltf) | `ltf/ltf_seed_0.ckpt` | `ltf_seed_0.ckpt` |

```bash
source scripts/env.sh
$SIMWAM_VENV/bin/python - <<'EOF'
import os
from huggingface_hub import snapshot_download
W = os.environ["GUIDEDWAM_WEIGHTS"]
snapshot_download("chenchenshi/DriveWAM", local_dir=f"{W}/DriveWAM", allow_patterns=["navsim/*"])
snapshot_download("robbyant/lingbot-va-base", local_dir=f"{W}/lingbot-va-base",
                  allow_patterns=["vae/*", "tokenizer/*", "text_encoder/*", "*.json"])
snapshot_download("H-EmbodVis/SimWAM", local_dir=f"{W}/SimWAM", allow_patterns=["weights/SimWAM.pt", "config.json"])
snapshot_download("tz2026/DriveLaW", local_dir=f"{W}/DriveLaW", allow_patterns=["DriveLaW-Act/*", "DriveLaW-Video/*", "config.json"],
                  ignore_patterns=["DriveLaW-Video/transformer/*"])
snapshot_download("autonomousvision/navsim_baselines", local_dir=f"{W}/navsim_baselines", allow_patterns=["ltf/ltf_seed_0.ckpt"])
EOF
ln -sfn $GUIDEDWAM_WEIGHTS/navsim_baselines/ltf/ltf_seed_0.ckpt $GUIDEDWAM_WEIGHTS/ltf_seed_0.ckpt
```

SimWAM's Wan2.2 backbone, and the text embedding of its fixed driving prompt, are prepared inside the SimWAM repo.
Link the weights first, so that `checkpoints/` points at `$GUIDEDWAM_WEIGHTS/simwam_checkpoints`:

```bash
mkdir -p $GUIDEDWAM_WEIGHTS/simwam_checkpoints && bash scripts/link_weights.sh
cd third_party/simwam && source $SIMWAM_VENV/bin/activate
bash scripts/model_prepare.sh          # Wan2.2-TI2V-5B (ModelScope) + ActionDiT backbone -> checkpoints/
bash scripts/precomput_text_embed.sh   # -> data/text_embeds_cache/navsim/*.t5_len256.wan22ti2v5b.pt
```

At inference SimWAM loads the Wan2.2 video DiT and VAE plus the ActionDiT backbone, then overwrites the video and
action experts with `SimWAM.pt`. It loads no text encoder (`load_text_encoder: false`), so the umt5-xxl encoder is
only used once, to build the prompt embedding.

`scripts/link_weights.sh` links the weights and the dataset into the paths the baselines expect:
`third_party/drivewam/weights/`, `third_party/simwam/{weights,checkpoints,data/navsim}`. DriveLaW reads
`$DRIVELAW_WEIGHTS` through `configs/drivelaw_eval_navsim.yaml`.

## 5. Check

```bash
source scripts/env.sh
for e in $DRIVEWAM_VENV $SIMWAM_VENV $DRIVELAW_VENV $FIELDS_VENV; do
  $e/bin/python -c "import guidedwam.energies.navhard_fp, navsim; print(navsim.__file__)"; done
```

Each line should print a path under `third_party/navsim/`.
