"""DriveWAM adapter for the guided sampler: normalized delta latents (B, C=30, F, N=8, 1) -> metric poses.

Conventions of DriveWAM's ``FlowMatchScheduler`` (action_snr_shift=1): x_sigma = (1-s)*x0 + s*n, v = n - x0, so
x0_hat = x - sigma * v_hat; with shift 1 the sigmas are linspace(1, 0, N+1)[:-1] and the window applies to raw sigma.
Only channels 0-2 = (dx, dy, dheading) are real: plain differences of consecutive poses in the CURRENT ego frame,
quantile min-max normalized to [-1, 1]; the inverse is the eval-side denorm + per-channel cumsum
(``NavSimEvaluator.actions_to_trajectory``), which is what PDM scores.
"""
from __future__ import annotations

import functools
import os
from typing import Sequence

import torch

from guidedwam.energies.kinematic import (ACTION_DT, ACTION_HORIZON, BrakeTargetEnergy, ComfortEnergy,  # noqa: F401
                                          TargetSpeedEnergy, build_brake_target, build_brake_target_from_speed,
                                          speeds_from_poses)
from guidedwam.sampler import guidance as _guidance
from guidedwam.sampler.guidance import JAC_DIAG, STATS, TIMING, TRACE, GuidanceSchedule  # noqa: F401

# navsim_cfg.norm_stat — (dx, dy, dheading) 1%/99% quantiles.
NAVSIM_Q01 = (-0.01582209, -1.76743735, -0.18790549)
NAVSIM_Q99 = (6.15755262, 2.46998216, 0.18174374)


def denorm_deltas(normalized: torch.Tensor,
                  q01: Sequence[float] = NAVSIM_Q01,
                  q99: Sequence[float] = NAVSIM_Q99) -> torch.Tensor:
    """(..., 3) normalized deltas -> metric deltas; differentiable affine.

    Verbatim eval-side inverse (``NavSimEvaluator.actions_to_trajectory``):
    delta = (a + 1) / 2 * (q99 - q01) + q01.
    """
    lo = torch.as_tensor(q01, device=normalized.device, dtype=normalized.dtype)
    hi = torch.as_tensor(q99, device=normalized.device, dtype=normalized.dtype)
    return (normalized + 1.0) / 2.0 * (hi - lo) + lo


def norm_deltas(deltas: torch.Tensor,
                q01: Sequence[float] = NAVSIM_Q01,
                q99: Sequence[float] = NAVSIM_Q99) -> torch.Tensor:
    """Dataset-side normalization (``NavSimEpisodeDataset._normalize_actions``):
    a = (delta - q01) / (q99 - q01 + 1e-6) * 2 - 1.  Note the 1e-6 that the
    eval-side denorm does NOT have (documented asymmetry, ~2e-7 relative)."""
    lo = torch.as_tensor(q01, device=deltas.device, dtype=deltas.dtype)
    hi = torch.as_tensor(q99, device=deltas.device, dtype=deltas.dtype)
    return (deltas - lo) / (hi - lo + 1e-6) * 2.0 - 1.0


def rollout_cumsum(deltas: torch.Tensor) -> torch.Tensor:
    """(B, T, 3) metric deltas -> (B, T, 3) absolute poses, per-channel cumsum.

    EXACTLY what DriveWAM's own ``actions_to_trajectory`` does (deltas are
    frame-0 ego-frame increments, heading included).  This is the conversion
    used for guidance so the energy sees the very trajectory PDM will score."""
    return torch.cumsum(deltas, dim=-2)


def rollout_se2(deltas: torch.Tensor) -> torch.Tensor:
    """(B, T, 3) BODY-FRAME SE(2) deltas -> (B, T, 3) absolute poses.

    Cumulative SE(2) composition: heading integrates dyaw; each step's local
    (dx, dy) is rotated by the heading at the START of the step before being
    accumulated:  p_k = p_{k-1} + R(theta_{k-1}) @ d_k,  theta_k = sum dyaw.
    Fully differentiable (cumsum + trig).  NOT the DriveWAM convention (see
    module docstring) — provided and unit-tested for local-delta models."""
    dxy = deltas[..., :2]
    dyaw = deltas[..., 2]
    yaw = torch.cumsum(dyaw, dim=-1)
    yaw_prev = yaw - dyaw                       # heading before step k
    c, s = torch.cos(yaw_prev), torch.sin(yaw_prev)
    gx = c * dxy[..., 0] - s * dxy[..., 1]
    gy = s * dxy[..., 0] + c * dxy[..., 1]
    xy = torch.cumsum(torch.stack([gx, gy], dim=-1), dim=-2)
    return torch.cat([xy, yaw.unsqueeze(-1)], dim=-1)


class DriveWamActionAdapter:
    """Differentiable (B, C, F, N, 1) action latents -> (B, N, 3) metric poses.

    Mirrors ``NavSimEvaluator.actions_to_trajectory`` step by step:
    pick ``channel_ids`` at chunk ``chunk_index`` -> quantile denorm ->
    delta rollout -> (B, N, 3) with columns (x, y, heading).  Default
    ``chunk_index=-1`` selects the future chunk both for the rollout tensor
    (F=1) and the concatenated 2-chunk tensor (F=2, future chunk last)."""

    def __init__(self,
                 q01: Sequence[float] = NAVSIM_Q01,
                 q99: Sequence[float] = NAVSIM_Q99,
                 channel_ids: Sequence[int] = (0, 1, 2),
                 chunk_index: int = -1,
                 rollout: str = "cumsum"):
        self.q01, self.q99 = tuple(q01), tuple(q99)
        self.channel_ids = list(channel_ids)
        self.chunk_index = chunk_index
        if rollout not in ("cumsum", "se2"):
            raise ValueError(f"unknown rollout '{rollout}'")
        self.rollout = rollout

    def metric_deltas(self, x: torch.Tensor) -> torch.Tensor:
        d = x[:, self.channel_ids, self.chunk_index, :, 0]      # (B, 3, N)
        d = d.permute(0, 2, 1)                                  # (B, N, 3)
        return denorm_deltas(d, self.q01, self.q99)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        d = self.metric_deltas(x)
        return rollout_cumsum(d) if self.rollout == "cumsum" else rollout_se2(d)


def shifted_sigma(sigma_raw: float, shift: float) -> float:
    """FlowMatchScheduler shift map s -> shift*s / (1 + (shift-1)*s).
    Identity for shift=1 (DriveWAM's action_snr_shift)."""
    return shift * sigma_raw / (1.0 + (shift - 1.0) * sigma_raw)


def action_sigma(scheduler, timestep) -> float:
    """Replicate FlowMatchScheduler.step's sigma lookup for a timestep."""
    if isinstance(timestep, torch.Tensor):
        timestep = timestep.cpu()
    timestep_id = torch.argmin((scheduler.timesteps - timestep).abs())
    return float(scheduler.sigmas[timestep_id])


@functools.wraps(_guidance.guided_step)
def guided_step(*args, to_metric=None, debug_tag="DRIVEWAM_GUIDANCE_DEBUG", **kwargs):
    # GUIDANCE_NORMALIZE=1: per-energy gradient normalization for the stabilization ablation (off in the paper runs)
    kwargs.setdefault("normalize", os.environ.get("GUIDANCE_NORMALIZE", "0") == "1")
    return _guidance.guided_step(*args, to_metric=to_metric or DriveWamActionAdapter(), debug_tag=debug_tag, **kwargs)
