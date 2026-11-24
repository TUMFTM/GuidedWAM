# Third-party code

The four directories here are copies of other projects, each under its own license. The license file of each project
is kept in its directory. The changes made for GuidedWAM are listed below.

| directory | project | source | base version | license |
|---|---|---|---|---|
| `drivewam/` | DriveWAM | release of the DriveWAM authors | commit `abbefb4` | Apache License 2.0 (`drivewam/LICENSE.txt`) |
| `simwam/` | SimWAM | <https://github.com/H-EmbodVis/SimWAM> | commit `68b426c` | MIT License (`simwam/LICENSE`) |
| `drivelaw/` | DriveLaW, planner part `DriveLaW-Act` only | <https://github.com/xiaomi-research/drivelaw> | commit `243e0e4` | Apache License 2.0 (`drivelaw/LICENSE`) |
| `navsim/` | NAVSIM devkit | <https://github.com/autonomousvision/navsim>, as bundled with SimWAM | v2 | Apache License 2.0 (`navsim/LICENSE`) |

## Changes

**DriveWAM**
- `src/navsim/guidance.py` (new): adapter to the guided sampler in `guidedwam.sampler`.
- `src/navsim/eval.py`, `src/modules/model.py`: optional energies, guidance schedule and velocity bound in the action
  denoising loop. Without energies the sampler is unchanged.
- `src/configs/navsim_cfg.py`, `.gitignore`: path and ignore settings.
- `tests/test_guidance_cpu.py` (new): unit tests of the guidance hook.

**SimWAM**
- `src/simwam/guidance.py` (new): adapter to the guided sampler in `guidedwam.sampler`.
- `src/simwam/models/wan22/simwam.py`: optional energies, guidance schedule and velocity bound in `infer_action`.
- `experiments/navsim/predict_navhard.py`, `experiments/navsim/eval_navsim.py`, `configs/sim_navsim.yaml`: guidance
  arms, window, seed and scene-subset options.
- `tests/test_guidance.py` (new): unit tests of the guidance hook.

**DriveLaW** (only `DriveLaW-Act` is included; the video model is not needed to run the planner code here)
- `navsim/agents/videodrive/video_models/pipeline/pipeline_ltx_condition.py`,
  `navsim/agents/videodrive/videodrive_agent.py`: optional energies, guidance schedule and velocity bound in the
  action sampler.
- `navsim/agents/npy_trajectory_agent.py`, `navsim/planning/script/config/common/agent/npy_trajectory_agent.yaml`
  (new): agent that scores stored trajectories.

**NAVSIM devkit**
- Added from the official devkit: the TransFuser agent (`navsim/agents/transfuser`), metric caching for NAVSIM v2
  (`navsim/planning/metric_caching`, the PDM planner files under `navsim/planning/simulation`), scenario-builder and
  visualization modules, and the navtrain scene filter.
- `scripts/evaluation/run_npy_trajectory_agent_pdm_score_evaluation.sh`: runs from the repository root.
