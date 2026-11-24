"""Kinematic energies on the metric ego-frame trajectory (B, T, 3) = (x, y, heading), 0.5 s spacing: braking toward
a target, a target speed, and longitudinal comfort. Speeds come from finite differences of the poses."""
from __future__ import annotations

import os

import torch

ACTION_DT = 0.5   # s between the 8 predicted poses (4 s horizon @ 2 Hz)
ACTION_HORIZON = 8


def speeds_from_poses(ego: torch.Tensor, dt: float = ACTION_DT) -> torch.Tensor:
    """Finite-difference speeds (B, T) from absolute poses (B, T, 3).

    Poses are absolute in the CURRENT ego frame, so the predecessor of step 0
    is the origin: v_0 = ||xy_0||/dt, v_k = ||xy_k - xy_{k-1}||/dt.  The +1e-9
    keeps the norm's gradient finite at zero displacement."""
    xy = ego[..., :2]
    prev = torch.cat([torch.zeros_like(xy[..., :1, :]), xy[..., :-1, :]], dim=-2)
    seg = ((xy - prev).pow(2).sum(-1) + 1e-9).sqrt()
    return seg / dt


class BrakeTargetEnergy:
    """MSE attraction toward a (braked) target trajectory.

    E = w_xy*mean_t ||xy - xy*||^2 + w_v*mean_t (v - v*)^2 with v the
    finite-difference speed of the plan and v* that of the target — steering
    displacement and speed together is what makes the plan read as braking
    (diwm TrajectoryTargetEnergy semantics)."""

    def __init__(self, target: torch.Tensor, w_xy: float = 1.0, w_v: float = 1.0,
                 dt: float = ACTION_DT):
        self.tgt = target.detach().float()          # (T, 3) or (B, T, 3)
        self.w_xy, self.w_v, self.dt = w_xy, w_v, dt
        self.tgt_v = speeds_from_poses(self.tgt, dt)

    def __call__(self, ego: torch.Tensor) -> torch.Tensor:
        tgt = self.tgt.to(device=ego.device, dtype=ego.dtype)
        exy = ((ego[..., :2] - tgt[..., :2]) ** 2).sum(-1).mean(-1)
        v = speeds_from_poses(ego, self.dt)
        ev = ((v - self.tgt_v.to(device=ego.device, dtype=ego.dtype)) ** 2).mean(-1)
        return self.w_xy * exy + self.w_v * ev


class TargetSpeedEnergy:
    """Pull the plan's finite-difference speed profile toward ``v_target``."""

    def __init__(self, v_target: float, weight: float = 1.0, dt: float = ACTION_DT):
        self.v, self.w, self.dt = float(v_target), weight, dt

    def __call__(self, ego: torch.Tensor) -> torch.Tensor:
        v = speeds_from_poses(ego, self.dt)
        return self.w * ((v - self.v) ** 2).mean(dim=-1)


class SpeedCapEnergy:
    """Hinge on the plan's finite-difference speed above a cap: E = w * sum_t max(0, v_t - v_max)^2."""

    def __init__(self, v_max: float, weight: float = 1.0, dt: float = ACTION_DT):
        self.v, self.w, self.dt = float(v_max), weight, dt

    def __call__(self, ego: torch.Tensor) -> torch.Tensor:
        v = speeds_from_poses(ego, self.dt)
        return self.w * torch.relu(v - self.v).pow(2).sum(dim=-1)


def with_speed_cap(energies):
    """Append a SpeedCapEnergy to an arm's energies when env SPEED_CAP [m/s] is set (weight SPEED_CAP_W)."""
    cap = os.environ.get("SPEED_CAP")
    if not cap:
        return energies
    en = SpeedCapEnergy(float(cap), weight=float(os.environ.get("SPEED_CAP_W", "1.0")))
    if os.environ.get("SPEED_CAP_PRECOND", "0") == "1":   # for absolute-pose action spaces (SimWAM, DriveLaW)
        en.precondition = "cumsum"
    return list(energies or []) + [en]


class ComfortEnergy:
    """Penalize longitudinal accel/jerk beyond comfort bounds."""

    def __init__(self, dt: float = ACTION_DT, a_max: float = 2.5,
                 j_max: float = 4.0, weight: float = 1.0):
        self.dt, self.a_max, self.j_max, self.w = dt, a_max, j_max, weight

    def __call__(self, ego: torch.Tensor) -> torch.Tensor:
        v = speeds_from_poses(ego, self.dt)
        a = (v[..., 1:] - v[..., :-1]) / self.dt
        j = (a[..., 1:] - a[..., :-1]) / self.dt
        ea = torch.relu(a.abs() - self.a_max) ** 2
        ej = torch.relu(j.abs() - self.j_max) ** 2
        return self.w * (ea.mean(dim=-1) + ej.mean(dim=-1))


def build_brake_target(plan: torch.Tensor, dt: float = ACTION_DT,
                       brake_to: float = 0.0, v_floor: float = 0.0) -> torch.Tensor:
    """Same-path speed re-profile of a METRIC plan (B, T, 3) -> (B, T, 3).

    Keeps the plan's path DIRECTION (per-step heading from the displacement,
    origin-prepended) and re-profiles speed linearly from the plan's initial
    speed to ``brake_to``, then re-integrates xy from the origin — a braked
    target actually covers less ground."""
    if plan.ndim == 2:
        plan = plan.unsqueeze(0)
    B, T, _ = plan.shape
    device, dtype = plan.device, plan.dtype
    xy = plan[..., :2]
    dt = max(float(dt), 1e-3)

    prev = torch.cat([torch.zeros_like(xy[:, :1]), xy[:, :-1]], dim=1)
    d = xy - prev                                                  # (B, T, 2)
    seg = torch.linalg.norm(d, dim=-1)                             # (B, T)
    head = torch.atan2(d[..., 1], d[..., 0])                       # (B, T)
    v_plan = seg / dt                                              # (B, T) m/s
    v0 = v_plan[:, : min(3, T)].mean(dim=1, keepdim=True).clamp(min=0.0)

    t_idx = torch.arange(T, device=device, dtype=dtype)[None, :]   # (1, T)
    frac = (1.0 - t_idx / max(T - 1, 1)).clamp(min=0.0)            # 1 -> 0
    v_tgt = ((v0 - brake_to) * frac + brake_to).clamp(min=v_floor)

    step = (v_tgt * dt)[..., None] * torch.stack(
        [torch.cos(head), torch.sin(head)], dim=-1
    )                                                              # (B, T, 2)
    xy_tgt = torch.cumsum(step, dim=1)                             # origin-anchored
    return torch.stack([xy_tgt[..., 0], xy_tgt[..., 1], head], dim=-1)


def build_brake_target_from_speed(v0: float, horizon: int = ACTION_HORIZON,
                                  dt: float = ACTION_DT,
                                  brake_to: float = 0.0) -> torch.Tensor:
    """Straight-line braking target (T, 3) from the measured ego speed alone."""
    t_idx = torch.arange(horizon, dtype=torch.float32)
    frac = (1.0 - t_idx / max(horizon - 1, 1)).clamp(min=0.0)
    v_tgt = (float(v0) - brake_to) * frac + brake_to
    x = torch.cumsum(v_tgt * dt, dim=0)
    return torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=-1)
