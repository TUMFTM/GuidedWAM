# GuidedWAM

Training-free energy guidance for frozen world action models (WAMs) in autonomous driving. Each driving objective is
a differentiable energy on the sampler's one-step clean estimate of the action stream. Its gradient is added to the
frozen model's velocity at every denoising step. No backpropagation goes through the network. The guidance is
stabilised by a trapezoidal window over the noise level and a velocity-bounded correction. Several energies are
combined as a weighted sum of their gradients. Per-energy gradient normalisation is available as an option and is not
used for the results of the paper.

Paper: [arXiv:2611.07912](https://arxiv.org/abs/2611.07912), WACV 2027 Workshop on World Action Models for Autonomous
Driving. Project page: <https://tumftm.github.io/GuidedWAM/> (source in [docs/index.html](docs/index.html)).

The guidance is evaluated on three publicly released WAMs on the NAVSIM v2 navhard split (EPDMS):

| model | stock (3 seeds) | + energy guidance (camera-only drivable area, 3 seeds) | gain |
|---|---|---|---|
| DriveLaW | 29.8 ± 0.3 | 33.6 ± 0.4 | +3.8 |
| DriveWAM | 35.5 ± 0.7 | 37.8 ± 0.8 | +2.3 |
| SimWAM | 37.3 ± 0.6 | 39.6 ± 0.5 | +2.2 |

- [docs/installation.md](docs/installation.md): environments, NAVSIM data and public weights
- [docs/run.md](docs/run.md): fields, one script per paper table and figure, single runs
- [docs/energies.md](docs/energies.md): exact form of every energy, the guided sampler and the camera-only road-edge field

## Repository

| path | content |
|---|---|
| `third_party/{drivewam,simwam,drivelaw}` | copies of the three baselines with the guidance hook in each sampler ([third_party/README.md](third_party/README.md)) |
| `third_party/navsim` | NAVSIM v2 devkit used for metric caching and scoring |
| `guidedwam/energies` | the energies (library) |
| `guidedwam/sampler` | guided sampler step and schedule |
| `scripts/` | fields, predictions, scoring, ablations, figures, and `scripts/paper/` per table and figure |
| `envs/` | lock files of the four Python environments |

## License

GuidedWAM is released under the Apache License 2.0 ([LICENSE](LICENSE)).

The code under `third_party/` belongs to its authors and keeps its own license: DriveWAM, DriveLaW and the NAVSIM
devkit under the Apache License 2.0, SimWAM under the MIT License. [third_party/README.md](third_party/README.md) lists
the sources and the changes made for this work.
