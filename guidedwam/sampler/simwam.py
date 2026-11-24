"""SimWAM adapter for the guided sampler: normalized absolute poses (B, T=8, 3) -> metric poses.

Conventions of SimWAM's ``WanContinuousFlowMatchScheduler``: x_sigma = (1-sigma)*x_data + sigma*noise,
v = noise - x_data, x0_hat = x_sigma - sigma*v_hat. The model denoises min-max normalized absolute ego-frame poses
(``trajectory_mode='absolute'``, ``normalize_action=true``); ``denorm_odo`` replicates
``NavSimVideoDataset.denorm_odo`` verbatim.
"""
from __future__ import annotations

import functools

import torch

from guidedwam.energies.kinematic import (ACTION_DT, BrakeTargetEnergy, ComfortEnergy, TargetSpeedEnergy,  # noqa: F401
                                          build_brake_target, build_brake_target_from_speed, speeds_from_poses)
from guidedwam.sampler import guidance as _guidance
from guidedwam.sampler.guidance import JAC_DIAG, STATS, TIMING, TRACE, GuidanceSchedule  # noqa: F401


def denorm_odo(normalized_trajectory: torch.Tensor) -> torch.Tensor:
    """Min-max denormalization to metric ego frame; differentiable affine."""
    x = (normalized_trajectory[..., 0:1] + 1) / 2 * 66.74 - 1.57
    y = (normalized_trajectory[..., 1:2] + 1) / 2 * 42 - 19.68
    heading = (normalized_trajectory[..., 2:3] + 1) / 2 * 3.53 - 1.67
    return torch.cat([x, y, heading], dim=-1)


def norm_odo(trajectory: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`denorm_odo` (used to build targets, tests)."""
    x = 2 * (trajectory[..., 0:1] + 1.57) / 66.74 - 1
    y = 2 * (trajectory[..., 1:2] + 19.68) / 42 - 1
    heading = 2 * (trajectory[..., 2:3] + 1.67) / 3.53 - 1
    return torch.cat([x, y, heading], dim=-1)


@functools.wraps(_guidance.guided_step)
def guided_step(pred_v, latents, sigma, energies, lam, clip_rel=12.0, denorm=denorm_odo,
                debug_tag="SIMWAM_GUIDANCE_DEBUG", **kwargs):
    return _guidance.guided_step(pred_v, latents, sigma, energies, lam, clip_rel, to_metric=denorm,
                                 debug_tag=debug_tag, **kwargs)
