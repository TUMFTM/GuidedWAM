"""Sampling-time energy guidance for SimWAM's action denoising loop.

The guided sampler lives in GuidedWAM (``guidedwam.sampler``, on PYTHONPATH via GuidedWAM's scripts/env.sh): the
model-agnostic step and window in ``guidedwam.sampler.guidance``, SimWAM's latent -> metric map ``denorm_odo``
(min-max normalized absolute poses, constants of ``NavSimVideoDataset``) in ``guidedwam.sampler.simwam``. This
module re-exports them under the names the model and the NAVSIM scripts use.
"""
from guidedwam.sampler.simwam import (  # noqa: F401
    ACTION_DT, JAC_DIAG, STATS, TIMING, TRACE,
    BrakeTargetEnergy, ComfortEnergy, GuidanceSchedule, TargetSpeedEnergy,
    build_brake_target, build_brake_target_from_speed, denorm_odo, guided_step, norm_odo, speeds_from_poses,
)
