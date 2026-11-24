"""CPU unit tests for DriveWAM sampling-time energy guidance.

Run from the repo root with the scratch venv:
    .venv-cpu/bin/python -m pytest tests/test_guidance_cpu.py -v

No GPU, no model weights: the transformer is replaced by a mock velocity field
where needed; the ACTUAL FlowMatchScheduler from src/utils/scheduler.py and the
ACTUAL inference schedule construction from src/navsim/eval.py
(_prepare_timesteps semantics: shift=1.0, sigma_min=0.0, extra_one_step=True,
timesteps padded with a trailing 0, last iteration skipped) are used verbatim.
"""
import importlib.util
import math
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO_ROOT)

from src.navsim.guidance import (  # noqa: E402
    ACTION_DT,
    NAVSIM_Q01,
    NAVSIM_Q99,
    BrakeTargetEnergy,
    DriveWamActionAdapter,
    GuidanceSchedule,
    action_sigma,
    build_brake_target,
    denorm_deltas,
    guided_step,
    norm_deltas,
    rollout_cumsum,
    rollout_se2,
    shifted_sigma,
    speeds_from_poses,
)

# Import FlowMatchScheduler straight from its file (src/utils/__init__ pulls in
# extra deps we don't need on CPU).
_spec = importlib.util.spec_from_file_location(
    "flow_sched", os.path.join(REPO_ROOT, "src", "utils", "scheduler.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FlowMatchScheduler = _mod.FlowMatchScheduler

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. SE(2) rollout correctness (hand-computed 3-step case incl. a turn)
# ---------------------------------------------------------------------------

def test_se2_rollout_hand_case():
    # step 1: forward 1 m then turn left 90 deg
    # step 2: forward 1 m (now facing +y)
    # step 3: left 1 m in body frame (= -x global), turn right 90 deg
    d = torch.tensor([[[1.0, 0.0, math.pi / 2],
                       [1.0, 0.0, 0.0],
                       [0.0, 1.0, -math.pi / 2]]])
    out = rollout_se2(d)
    expected = torch.tensor([[[1.0, 0.0, math.pi / 2],
                              [1.0, 1.0, math.pi / 2],
                              [0.0, 1.0, 0.0]]])
    assert torch.allclose(out, expected, atol=1e-6), f"{out} != {expected}"


def test_se2_reduces_to_cumsum_without_rotation():
    d = torch.randn(2, 8, 3)
    d[..., 2] = 0.0  # no yaw -> body frame == global frame
    assert torch.allclose(rollout_se2(d)[..., :2], rollout_cumsum(d)[..., :2], atol=1e-6)


def test_rollouts_are_differentiable():
    d = torch.randn(2, 8, 3, requires_grad=True)
    for fn in (rollout_cumsum, rollout_se2):
        e = (fn(d) ** 2).sum()
        (g,) = torch.autograd.grad(e, d, retain_graph=False)
        assert torch.isfinite(g).all() and g.abs().sum() > 0


# ---------------------------------------------------------------------------
# 2. Adapter replicates NavSimEvaluator.actions_to_trajectory verbatim
# ---------------------------------------------------------------------------

def _repo_actions_to_trajectory(actions):
    """Verbatim re-implementation of NavSimEvaluator.actions_to_trajectory."""
    deltas = actions[:, [0, 1, 2], 1, :, 0].float()                    # (B, 3, 8)
    q01 = torch.tensor(NAVSIM_Q01, dtype=deltas.dtype)
    q99 = torch.tensor(NAVSIM_Q99, dtype=deltas.dtype)
    deltas = (deltas + 1.0) / 2.0 * (q99 - q01)[:, None] + q01[:, None]
    return torch.cumsum(deltas, dim=-1)                                # (B, 3, 8)


def test_adapter_matches_repo_conversion():
    x = torch.rand(2, 30, 2, 8, 1) * 2 - 1
    repo = _repo_actions_to_trajectory(x)                              # (B, 3, 8)
    ours = DriveWamActionAdapter(chunk_index=-1)(x)                    # (B, 8, 3)
    assert torch.allclose(ours.permute(0, 2, 1), repo, atol=1e-6)


def test_norm_denorm_roundtrip_and_documented_asymmetry():
    d = torch.tensor([[3.0, -0.5, 0.05], [0.1, 1.2, -0.1]])
    rt = denorm_deltas(norm_deltas(d))
    # Dataset norm divides by (q99-q01+1e-6) but eval denorm multiplies by
    # (q99-q01): round trip is off by a relative ~1.2e-7 — tiny but nonzero.
    assert torch.allclose(rt, d, atol=1e-4)
    assert not torch.equal(rt, d)


# ---------------------------------------------------------------------------
# 3. Guidance window
# ---------------------------------------------------------------------------

def test_window_zeros_and_shape():
    sch = GuidanceSchedule(lam_max=2.0)
    assert sch(1.0) == 0.0 and sch(0.95) == 0.0 and sch(0.96) == 0.0
    assert sch(0.05) == 0.0 and sch(0.01) == 0.0 and sch(0.0) == 0.0
    assert sch(0.8) == 2.0 and sch(0.5) == 2.0 and sch(0.3) == 2.0
    assert 0.0 < sch(0.9) < 2.0 and 0.0 < sch(0.1) < 2.0
    # ramp linearity
    assert abs(sch(0.875) - 2.0 * 0.5) < 1e-9
    assert abs(sch(0.175) - 2.0 * 0.5) < 1e-9
    assert GuidanceSchedule(mode="off", lam_max=2.0)(0.5) == 0.0
    assert GuidanceSchedule(mode="constant", lam_max=2.0)(0.99) == 2.0
    # shift map is identity for DriveWAM's action_snr_shift = 1.0
    for s in (0.0, 0.3, 0.77, 1.0):
        assert abs(shifted_sigma(s, 1.0) - s) < 1e-12


# ---------------------------------------------------------------------------
# 4. Off-mode identity and clip equality
# ---------------------------------------------------------------------------

def test_off_mode_returns_same_object():
    v = torch.randn(1, 30, 1, 8, 1, dtype=torch.bfloat16)
    x = torch.randn_like(v)
    assert guided_step(v, x, 0.5, [], 1.0) is v
    assert guided_step(v, x, 0.5, None, 1.0) is v
    tgt = build_brake_target(torch.randn(1, 8, 3))
    assert guided_step(v, x, 0.5, [BrakeTargetEnergy(tgt)], 0.0) is v


def test_clip_equality():
    torch.manual_seed(1)
    v = torch.randn(2, 30, 1, 8, 1)
    x = torch.randn_like(v)
    # Huge target -> huge gradient -> the clip must bind exactly.
    tgt = torch.zeros(8, 3)
    tgt[:, 0] = torch.linspace(1e4, 8e4, 8)
    clip_rel, lam = 12.0, 2.0
    out = guided_step(v, x, 0.5, [BrakeTargetEnergy(tgt)], lam, clip_rel=clip_rel)
    corr = (out - v).reshape(2, -1).norm(dim=1)
    vn = v.reshape(2, -1).norm(dim=1)
    assert torch.allclose(corr, clip_rel * vn, rtol=1e-4), (corr, clip_rel * vn)


def test_small_gradient_not_clipped_and_dtype_preserved():
    v = torch.randn(1, 30, 1, 8, 1, dtype=torch.bfloat16)
    x = torch.randn_like(v)
    tgt = build_brake_target(torch.randn(1, 8, 3))
    out = guided_step(v, x, 0.5, [BrakeTargetEnergy(tgt)], 1e-3)
    assert out.dtype == torch.bfloat16
    assert not torch.equal(out, v)  # active guidance does change something


# ---------------------------------------------------------------------------
# 5. The actual inference schedule, reconstructed on CPU
# ---------------------------------------------------------------------------

def _make_action_scheduler(num_steps):
    """Exactly NavSimEvaluator: FlowMatchScheduler(shift=action_snr_shift=1.0,
    sigma_min=0.0, extra_one_step=True) + _prepare_timesteps zero-padding."""
    sch = FlowMatchScheduler(shift=1.0, sigma_min=0.0, extra_one_step=True)
    sch.set_timesteps(num_steps)
    timesteps = torch.nn.functional.pad(sch.timesteps, (0, 1), mode="constant", value=0)
    return sch, timesteps


def test_reconstructed_schedule_matches_survey_facts():
    sch, timesteps = _make_action_scheduler(10)
    expect = torch.linspace(1.0, 0.0, 11)[:-1]
    assert torch.allclose(sch.sigmas, expect, atol=1e-6)          # 1.0 ... 0.1
    assert len(timesteps) == 11 and float(timesteps[-1]) == 0.0
    # action_sigma replicates step()'s argmin lookup
    for i, t in enumerate(timesteps[:-1]):
        assert abs(action_sigma(sch, t) - float(sch.sigmas[i])) < 1e-6


def _x_clean_and_mask():
    """A straight cruise plan: 3.0 m per 0.5 s step (6 m/s), no lateral/yaw,
    embedded into the (1, 30, 1, 8, 1) action tensor via the eval-side inverse
    normalization; mask replicates actions_mask (channels 0-2 active)."""
    clean_deltas = torch.zeros(1, 8, 3)
    clean_deltas[..., 0] = 3.0
    q01 = torch.tensor(NAVSIM_Q01)
    q99 = torch.tensor(NAVSIM_Q99)
    clean_norm = 2.0 * (clean_deltas - q01) / (q99 - q01) - 1.0
    x_clean = torch.zeros(1, 30, 1, 8, 1)
    x_clean[:, [0, 1, 2], 0, :, 0] = clean_norm[0].T
    mask = torch.zeros(1, 30, 1, 8, 1, dtype=torch.bool)
    mask[:, [0, 1, 2]] = True
    return x_clean, mask


def _mock_rollout(num_steps, energies=None, schedule=None, clip_rel=12.0,
                  seed=0, eps=0.1):
    """Replicates eval.py::_rollout_action_chunk's loop structure (actual
    FlowMatchScheduler, padded timesteps, last-step skip, mask re-zeroing)
    with a mock velocity field: the exact flow-matching velocity for a
    Gaussian data distribution N(x_clean, eps^2) —
        v(x, s) = (x - mu)/s,  mu = x_clean + w*(x - (1-s)*x_clean),
        w = (1-s)*eps^2 / ((1-s)^2*eps^2 + s^2)
    (eps=0 degenerates to the delta distribution, v = (x - x_clean)/s).
    Like a real denoiser, mu tracks the guided latent as sigma shrinks, so the
    model's own evolving plan (the UNCORRECTED x0-hat = mu) is a meaningful
    per-step readout.  Returns (final clean actions, per-step plan endpoint
    errors to the energy target, x_clean).
    """
    torch.manual_seed(seed)
    adapter = DriveWamActionAdapter()
    x_clean, mask = _x_clean_and_mask()

    sch, timesteps = _make_action_scheduler(num_steps)
    schedule = schedule or GuidanceSchedule(lam_max=0.2)

    actions = torch.randn(1, 30, 1, 8, 1)
    actions = torch.where(mask, actions, torch.zeros_like(actions))
    plan_endpoint_errs = []
    tgt = energies[0].tgt if energies else None

    for step_id, t in enumerate(timesteps):
        last_step = step_id == len(timesteps) - 1
        actions = torch.where(mask, actions, torch.zeros_like(actions))
        if last_step:
            continue                                  # eval.py: cache-commit only
        sigma = action_sigma(sch, t)
        w = (1 - sigma) * eps ** 2 / ((1 - sigma) ** 2 * eps ** 2 + sigma ** 2)
        mu = x_clean + w * (actions - (1 - sigma) * x_clean)
        pred = torch.where(mask, (actions - mu) / sigma, torch.zeros_like(actions))
        if energies:
            if tgt is not None:
                plan_endpoint_errs.append(
                    float((adapter(mu)[0, -1, :2] - tgt[..., -1, :2].reshape(-1)).norm()))
            lam = schedule(sigma)
            pred = guided_step(pred, actions, sigma, energies, lam,
                               clip_rel=clip_rel, to_metric=adapter, mask=mask)
        actions = sch.step(pred, t, actions)

    return torch.where(mask, actions, torch.zeros_like(actions)), plan_endpoint_errs, x_clean


@pytest.mark.parametrize("num_steps", [5, 10])
@pytest.mark.parametrize("lam", [0.1, 0.2])
def test_monotone_brake_approach_through_inference_schedule(num_steps, lam):
    # Unguided baseline (delta distribution): sampler must land on x_clean.
    final_u, _, x_clean = _mock_rollout(num_steps, energies=None, eps=0.0)
    assert torch.allclose(final_u, x_clean, atol=1e-5)

    adapter = DriveWamActionAdapter()
    plan = adapter(x_clean)                                        # (1, 8, 3)
    target = build_brake_target(plan, dt=ACTION_DT, brake_to=0.0)[0]
    energies = [BrakeTargetEnergy(target)]

    final_g, errs, _ = _mock_rollout(
        num_steps, energies=energies, schedule=GuidanceSchedule(lam_max=lam))

    # (a) monotone approach: the model's evolving plan endpoint error is
    #     non-increasing through the actual inference schedule.
    #     NOTE (calibration finding): DriveWAM's
    #     cumsum delta->absolute Jacobian amplifies the metric gradient by
    #     ~T/2 vs an affine-only action space (SimWAM), so effective lambdas
    #     are ~5-10x smaller here; lam >= 0.5 overshoots in this mock.
    assert len(errs) >= 2
    for a, b in zip(errs, errs[1:]):
        assert b <= a + 1e-6, f"plan endpoint error increased: {errs}"

    # (b) the final sampled plan is strictly closer to the brake target than
    #     the unguided plan, and slower.
    err_unguided = float((plan[0, -1, :2] - target[-1, :2]).norm())
    traj_g = adapter(final_g)
    err_guided = float((traj_g[0, -1, :2] - target[-1, :2]).norm())
    assert err_guided < err_unguided * 0.6, (err_guided, err_unguided)
    v_u = speeds_from_poses(plan).mean()
    v_g = speeds_from_poses(traj_g).mean()
    assert v_g < v_u


def test_off_mode_bitwise_identity_through_loop():
    """The injected code path with energies=None must be bit-identical to the
    stock loop (same RNG stream, same scheduler arithmetic)."""
    stock, _, _ = _mock_rollout(10, energies=None, seed=123)
    injected, _, _ = _mock_rollout(10, energies=[], seed=123)      # falsy -> gate closed
    assert torch.equal(stock, injected)
    # and guided_step itself is an identity pass-through (same object)
    v = torch.randn(1, 30, 1, 8, 1)
    assert guided_step(v, torch.randn_like(v), 0.7, [], 1.0) is v


def test_window_inactive_at_first_step_sigma_one():
    """sigma=1.0 (first inference step) lies outside the trapezoid: lambda=0,
    so guided_step must return the velocity unchanged (same object)."""
    sch = GuidanceSchedule()
    assert sch(1.0) == 0.0
    v = torch.randn(1, 30, 1, 8, 1)
    x = torch.randn_like(v)
    tgt = build_brake_target(torch.randn(1, 8, 3))
    assert guided_step(v, x, 1.0, [BrakeTargetEnergy(tgt)], sch(1.0)) is v
