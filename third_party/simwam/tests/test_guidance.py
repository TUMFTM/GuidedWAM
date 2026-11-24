"""CPU unit tests for simwam.guidance (pure torch, no model / no data needed).

    python tests/test_guidance.py

A synthetic rectified-flow denoiser with a KNOWN clean trajectory drives a
10-step schedule that mirrors SimWAM's `build_inference_schedule` (shift=5),
and the tests assert:
  (a) guided_step moves the x̂₀ endpoint toward a brake target monotonically,
  (b) off mode (energies=None / lam=0) is bit-identical to the stock loop,
  (c) the clip caps the applied correction at clip_rel * ||v||,
  (d) the trapezoidal window zeroes guidance outside its sigma range.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simwam.guidance import (  # noqa: E402
    ACTION_DT,
    BrakeTargetEnergy,
    ComfortEnergy,
    GuidanceSchedule,
    TargetSpeedEnergy,
    build_brake_target,
    build_brake_target_from_speed,
    denorm_odo,
    guided_step,
    norm_odo,
    speeds_from_poses,
)

T = 8  # SimWAM action horizon: 8 poses @ 0.5 s


def straight_plan(v0: float) -> torch.Tensor:
    """Constant-speed straight metric plan (1, T, 3), absolute ego-frame poses."""
    x = torch.arange(1, T + 1, dtype=torch.float32) * v0 * ACTION_DT
    return torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=-1)[None]


def simwam_schedule(n_steps: int, shift: float = 5.0) -> torch.Tensor:
    """Sigma grid of WanContinuousFlowMatchScheduler.build_inference_schedule."""
    u = torch.linspace(1.0, 0.0, n_steps + 1)
    return shift * u / (1.0 + (shift - 1.0) * u)


def run_loop(energies, lam_max: float, clip_rel: float = 12.0,
             n_steps: int = 10, seed: int = 0):
    """Euler loop with the exact injection-point semantics of `infer_action`.

    Synthetic denoiser: fixed noise n and clean sample x_data give the constant
    rectified-flow field v = n - x_data, so x̂₀ = x - sigma * v inherits any
    guidance deviation of the latents one-for-one (a perfectly self-correcting
    toy denoiser would erase it and test nothing).
    """
    torch.manual_seed(seed)
    x_data = norm_odo(straight_plan(v0=8.0))
    noise = torch.randn_like(x_data)
    schedule = GuidanceSchedule(lam_max=lam_max)
    sigmas = simwam_schedule(n_steps)
    x = noise.clone()
    x0_dists = []  # endpoint distance (m) of denorm(x̂₀) to the brake target
    for i in range(n_steps):
        sigma = float(sigmas[i])
        pred_v = noise - x_data
        if energies:  # the `infer_action` injection guard, verbatim semantics
            x0 = denorm_odo(x - sigma * pred_v)
            if isinstance(energies[0], BrakeTargetEnergy):
                tgt = energies[0].tgt
                x0_dists.append(float(torch.linalg.norm(x0[0, -1, :2] - tgt[0, -1, :2])))
            lam = schedule(sigma)
            pred_v = guided_step(pred_v, x, sigma, energies, lam, clip_rel)
        x = x + pred_v * (sigmas[i + 1] - sigmas[i])
    return x, x0_dists


if __name__ == "__main__":
    torch.manual_seed(0)

    # --- normalization roundtrip (constants must invert exactly) ------------
    plan = straight_plan(v0=8.0)
    assert torch.allclose(denorm_odo(norm_odo(plan)), plan, atol=1e-4)
    v = speeds_from_poses(plan)
    assert torch.allclose(v, torch.full_like(v, 8.0), atol=1e-4)
    print("[ok] denorm_odo/norm_odo roundtrip + finite-diff speeds")

    # --- brake target: same path, less ground, ramped speed -----------------
    target = build_brake_target(plan, brake_to=0.0)
    assert target.shape == plan.shape
    assert float(target[0, -1, 0]) < float(plan[0, -1, 0])          # covers less ground
    v_tgt = speeds_from_poses(target)
    assert abs(float(v_tgt[0, 0]) - 8.0) < 1e-3                     # starts at plan speed
    assert float(v_tgt[0, -1]) < 0.1                                # ends ~stopped
    straight_tgt = build_brake_target_from_speed(8.0, horizon=T)
    assert torch.allclose(straight_tgt[:, 1], torch.zeros(T))       # straight line
    print(f"[ok] brake target  plan_end={float(plan[0, -1, 0]):.1f}m -> "
          f"target_end={float(target[0, -1, 0]):.1f}m, v: 8.0 -> {float(v_tgt[0, -1]):.2f}")

    # --- (d) window zeroes guidance outside its sigma range -----------------
    sched = GuidanceSchedule(lam_max=2.0)
    assert sched(1.0) == 0.0 and sched(0.98) == 0.0                 # near sigma=1
    assert sched(0.03) == 0.0 and sched(0.0) == 0.0                 # near sigma=0
    assert sched(0.5) == 2.0 and sched(0.8) == 2.0 and sched(0.3) == 2.0
    assert 0.0 < sched(0.9) < 2.0 and 0.0 < sched(0.1) < 2.0        # ramps
    assert abs(sched(0.9) - 2.0 * (0.95 - 0.9) / (0.95 - 0.8)) < 1e-9
    assert GuidanceSchedule(mode="off", lam_max=2.0)(0.5) == 0.0
    assert GuidanceSchedule(mode="constant", lam_max=2.0)(0.99) == 2.0
    print("[ok] trapezoidal window (zero outside, lam_max inside, linear ramps)")

    # in-loop: an energy under a zero window must leave the sample untouched
    en = [BrakeTargetEnergy(target, w_v=0.0)]
    x_win, _ = run_loop(en, lam_max=0.0)
    x_off, _ = run_loop(None, lam_max=0.0)
    assert torch.equal(x_win, x_off)
    print("[ok] lam=0 window in-loop: guided path bitwise equals unguided")

    # --- (b) off mode is bit-identical to no hook ---------------------------
    x_data = norm_odo(straight_plan(v0=8.0))
    noise = torch.randn_like(x_data)
    pred_v = noise - x_data
    assert guided_step(pred_v, noise, 0.5, [], 1.0) is pred_v       # empty energies
    assert guided_step(pred_v, noise, 0.5, en, 0.0) is pred_v       # lam = 0
    x_a, _ = run_loop(None, lam_max=1.0, seed=7)
    x_b, _ = run_loop([], lam_max=1.0, seed=7)
    assert torch.equal(x_a, x_b)
    print("[ok] off mode: guided_step returns the identical tensor object")

    # --- (a) monotone approach to the brake target --------------------------
    x_fin, dists = run_loop([BrakeTargetEnergy(target, w_v=0.0)], lam_max=0.008)
    assert len(dists) == 10 and dists[0] > 1.0
    for d_prev, d_next in zip(dists, dists[1:]):
        assert d_next <= d_prev + 1e-6, (d_prev, d_next)
    assert dists[-1] < 0.5 * dists[0], dists
    # and the final SAMPLE endpoint ends nearer the brake target than unguided
    x_ung, _ = run_loop(None, lam_max=0.0)
    d_fin = float(torch.linalg.norm(denorm_odo(x_fin)[0, -1, :2] - target[0, -1, :2]))
    d_ung = float(torch.linalg.norm(denorm_odo(x_ung)[0, -1, :2] - target[0, -1, :2]))
    assert d_fin < d_ung
    print(f"[ok] monotone brake approach  x0-hat endpoint dist "
          f"{dists[0]:.2f}m -> {dists[-1]:.2f}m; sample {d_ung:.2f}m -> {d_fin:.2f}m")

    # --- (c) the clip caps the applied correction ---------------------------
    lam, clip_rel = 1.0, 12.0
    heavy = [BrakeTargetEnergy(target, w_xy=1e6, w_v=0.0)]
    v_guided = guided_step(pred_v, noise, 0.5, heavy, lam, clip_rel)
    corr = (v_guided - pred_v).reshape(1, -1).norm()
    cap = clip_rel * pred_v.reshape(1, -1).norm()
    assert float(corr) <= float(cap) * (1 + 1e-4)
    assert float(corr) >= float(cap) * 0.999                        # cap is active
    print(f"[ok] clip  ||lam*g||={float(corr):.3f} == clip_rel*||v||={float(cap):.3f}")

    # --- remaining energies: finite grads, sane signs -----------------------
    for energy in (ComfortEnergy(), TargetSpeedEnergy(0.0), BrakeTargetEnergy(target)):
        xg = norm_odo(plan).clone().requires_grad_(True)
        (g,) = torch.autograd.grad(energy(denorm_odo(xg)).sum(), xg)
        assert torch.isfinite(g).all()
    hard_brake = straight_plan(8.0).clone()
    hard_brake[0, 4:, 0] = hard_brake[0, 3, 0]                      # slam to 0 m/s
    assert float(ComfortEnergy()(hard_brake)) > 0.0 == float(ComfortEnergy()(plan))
    assert float(TargetSpeedEnergy(8.0)(plan)) < float(TargetSpeedEnergy(8.0)(target))
    print("[ok] comfort/target-speed energies: finite grads + sane ordering")

    print("all guidance tests passed")
