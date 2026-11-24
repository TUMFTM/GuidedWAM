"""Sampling-time energy guidance for DriveWAM's action denoising loop.

The guided sampler lives in GuidedWAM (``guidedwam.sampler``, on PYTHONPATH via GuidedWAM's scripts/env.sh): the
model-agnostic step and window in ``guidedwam.sampler.guidance``, DriveWAM's latent -> metric adapter (quantile
denorm + per-channel cumsum of the (dx, dy, dheading) channels) in ``guidedwam.sampler.drivewam``. This module
re-exports them under the names ``src.navsim.eval`` uses.

Off mode is byte-identical to stock: ``guided_step`` returns ``pred_v`` unchanged (the same object) when no
energies are given or lambda <= 0, and the injection in ``eval.py`` is gated by ``if energies:``.
"""
from guidedwam.sampler.drivewam import (  # noqa: F401
    ACTION_DT, ACTION_HORIZON, JAC_DIAG, NAVSIM_Q01, NAVSIM_Q99, STATS, TIMING, TRACE,
    BrakeTargetEnergy, ComfortEnergy, DriveWamActionAdapter, GuidanceSchedule, TargetSpeedEnergy,
    action_sigma, build_brake_target, build_brake_target_from_speed, denorm_deltas, guided_step, norm_deltas,
    rollout_cumsum, rollout_se2, shifted_sigma, speeds_from_poses,
)
