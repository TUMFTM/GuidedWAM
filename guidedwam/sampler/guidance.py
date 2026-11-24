"""Sampling-time energy guidance on the one-step clean estimate (model-agnostic).

At each denoising step of a flow-matching sampler the clean estimate is formed from the current latents and the
predicted velocity, with the velocity DETACHED (no backprop through the denoiser):

    x0_hat = latents - sigma * v_hat,

the energies are evaluated on the METRIC ego-frame trajectory ``to_metric(x0_hat)`` of shape (B, T, 3) =
(x, y, heading), and g = grad_{x0_hat} E is added to the velocity as lambda(sigma) * g after a per-sample clip
that keeps ||lambda * g|| <= clip_rel * ||v_hat|| (velocity-bounded correction). lambda(sigma) is a window on the
noise level (``GuidanceSchedule``). Adding lambda * g to v_hat moves both the next latent and the implied x0_hat
along -grad E, since the Euler step x' = x + v * (sigma' - sigma) has sigma' < sigma.

Each WAM supplies its own ``to_metric`` (normalized action latents -> metric poses); see the sibling modules
``drivewam``, ``simwam`` and ``drivelaw``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

JAC_DIAG: list = []  # per-step JF-vs-Jacobian gradient norms + cosine (Jacobian runs)
STATS: list = []     # per-step guidance statistics when env GUIDANCE_STATS is set (caller flushes per scene)
TIMING: list = []    # per-step forward/guidance time + peak memory (the sampler fills it under GUIDANCE_STATS)
TRACE: list = []     # per-step metric x0_hat before/after the correction when env GUIDANCE_TRACE is set


@dataclass
class GuidanceSchedule:
    """lambda(sigma) window. ``asymmetric`` is the trapezoid: 0 for sigma >= s_start or sigma <= s_end, lam_max in
    [s_full_lo, s_full_hi], linear ramps between. The other shapes are the window ablation."""

    mode: str = "asymmetric"          # asymmetric | constant | off | uniform | canonical | early | late
    lam_max: float = 1.0
    s_start: float = 0.95
    s_full_hi: float = 0.8
    s_full_lo: float = 0.3
    s_end: float = 0.05
    scale: float = 1.0                # set by `calibrate` so the window-shape ablation arms match total weight

    def _trapezoid(self, sigma: float) -> float:
        if sigma >= self.s_start or sigma <= self.s_end:
            return 0.0
        if sigma > self.s_full_hi:
            return self.lam_max * (self.s_start - sigma) / (self.s_start - self.s_full_hi)
        if sigma < self.s_full_lo:
            return self.lam_max * (sigma - self.s_end) / (self.s_full_lo - self.s_end)
        return self.lam_max

    def _shape(self, sigma: float) -> float:
        """Unnormalised window shapes for the ablation; `scale` matches their total weight over the sampler's
        sigma grid to the trapezoid's."""
        if self.mode == "uniform":                   # same weight at every step, sigma=1 included
            return self.lam_max
        if self.mode == "canonical":                 # lam*(t) = gamma t/(1-t), t = noise level, cut at 0.95
            return self.lam_max * sigma / (1.0 - sigma) if sigma <= self.s_start else 0.0
        mid = 0.5 * (self.s_start + self.s_end)
        if self.mode == "early":                     # first half of the window (high noise)
            return self._trapezoid(sigma) if sigma >= mid else 0.0
        if self.mode == "late":                      # second half of the window (low noise)
            return self._trapezoid(sigma) if sigma < mid else 0.0
        raise ValueError(f"unknown schedule mode {self.mode}")

    def calibrate(self, grid) -> "GuidanceSchedule":
        """Rescale a non-trapezoid shape so sum_k lam(sigma_k) over `grid` equals the trapezoid's."""
        if self.mode in ("asymmetric", "constant", "off"):
            return self
        target = sum(self._trapezoid(float(s)) for s in grid)
        raw = sum(self._shape(float(s)) for s in grid)
        self.scale = target / raw if raw > 0 else 0.0
        return self

    def __call__(self, sigma: float) -> float:
        if self.mode == "off":
            return 0.0
        if self.mode == "constant":
            return self.lam_max
        if self.mode == "asymmetric":
            return self._trapezoid(sigma)
        return self.scale * self._shape(sigma)


def _precondition_cumsum(gm: torch.Tensor) -> torch.Tensor:
    """Displacement-space preconditioner for a metric gradient (B, T, 3) over absolute poses p_t = sum_{s<=t} d_s
    (origin fixed): descending in the displacements d gives the position direction C C^T dE/dp, with C the
    lower-triangular ones matrix, i.e. a reverse cumulative sum followed by a cumulative sum over time. For an energy
    of the displacements alone (the speed cap) the preconditioned Hessian is that of the displacements, which is
    diagonal, instead of the second-difference operator that makes the plain Euler step oscillate."""
    return torch.cumsum(torch.flip(torch.cumsum(torch.flip(gm, [-2]), -2), [-2]), -2)


def _energy_grad(en, ek: torch.Tensor, m: torch.Tensor, xh: torch.Tensor) -> torch.Tensor:
    """Latent gradient of one energy term ek = en(m).sum(), m = to_metric(xh). With en.precondition == 'cumsum' the
    metric gradient is preconditioned (``_precondition_cumsum``) before it is pulled back through to_metric."""
    if getattr(en, "precondition", None) == "cumsum":
        (gm,) = torch.autograd.grad(ek, m, retain_graph=True)
        (gk,) = torch.autograd.grad(m, xh, grad_outputs=_precondition_cumsum(gm), retain_graph=True)
        return gk
    (gk,) = torch.autograd.grad(ek, xh, retain_graph=True)
    return gk


def guided_step(
    pred_v: torch.Tensor,
    latents: torch.Tensor,
    sigma: float,
    energies: List[Callable[[torch.Tensor], torch.Tensor]],
    lam: float,
    clip_rel: float = 12.0,
    to_metric: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    mask: Optional[torch.Tensor] = None,
    grad_override: Optional[torch.Tensor] = None,
    normalize: bool = False,
    debug_tag: str = "WAM_GUIDANCE_DEBUG",
) -> torch.Tensor:
    """Return the energy-corrected velocity v_hat + lambda * g for one denoising step.

    ``to_metric`` maps x0_hat (the model's latent action tensor) to the metric trajectory; autograd chains the
    metric gradient back to the latent space. ``mask`` (bool, broadcastable to latents) zeroes the correction on
    inactive channels. ``grad_override`` replaces g by a caller-supplied gradient (the Jacobian form, backprop
    through the network); clip and injection stay the same. ``normalize=True`` divides each energy's gradient by
    its own per-sample L2 norm before summation (unit relative weights). Returns ``pred_v`` unchanged (the same
    object) when lam <= 0 or no energies are given. Math is float32 regardless of the model dtype; the correction
    is cast back to ``pred_v.dtype``."""
    assert to_metric is not None
    _trace = bool(os.environ.get("GUIDANCE_TRACE")) and bool(energies)
    if _trace:
        with torch.no_grad():
            _pre = to_metric(latents.detach().float() - float(sigma) * pred_v.detach().float())
        TRACE.append({"sigma": float(sigma), "lam": float(lam), "pre": _pre[0].cpu().tolist(), "post": None})
    if not energies or lam <= 0.0:
        return pred_v
    # callers may be inside torch.inference_mode() (DriveLaW's forward_test): leave it explicitly,
    # and clone so x0_hat is a normal (non-inference) tensor that can require grad.
    with torch.inference_mode(False), torch.enable_grad():
        xh = (latents.detach().float() - float(sigma) * pred_v.detach().float()).clone().requires_grad_(True)
        if normalize:
            m = to_metric(xh); gs = []; e = torch.zeros((), device=xh.device)
            for en in energies:
                ek = en(m).sum(); e = e + ek.detach()
                gk = _energy_grad(en, ek, m, xh)
                gk = gk.reshape(gk.shape[0], -1); gs.append(gk / gk.norm(dim=1, keepdim=True).clamp(min=1e-12))
            grad = sum(gs).reshape(xh.shape)
        elif any(getattr(en, "precondition", None) for en in energies):
            m = to_metric(xh); grad = torch.zeros_like(xh); e = torch.zeros((), device=xh.device)
            for en in energies:
                ek = en(m).sum(); e = e + ek.detach()
                grad = grad + _energy_grad(en, ek, m, xh)
        else:
            e = sum(en(to_metric(xh)) for en in energies).sum()
            (grad,) = torch.autograd.grad(e, xh)
    grad = grad.detach()
    _grad_jf = grad
    if grad_override is not None:
        grad = grad_override.detach().float()
        # diagnostics: how the through-network gradient relates to the Jacobian-free one
        _gj = grad.reshape(grad.shape[0], -1); _gf = _grad_jf.reshape(grad.shape[0], -1)
        JAC_DIAG.append({'g_jf': float(_gf[0].norm()), 'g_jac': float(_gj[0].norm()),
                         'cos': float((_gf[0] @ _gj[0]) / (_gf[0].norm() * _gj[0].norm() + 1e-12))})
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    if mask is not None:
        grad = grad * mask.to(dtype=grad.dtype, device=grad.device)
    B = grad.shape[0]
    gn = grad.reshape(B, -1).norm(dim=1).clamp(min=1e-12)
    vn = pred_v.detach().float().reshape(B, -1).norm(dim=1)
    cap = (clip_rel * vn / lam).clamp(min=1e-12)
    scale = torch.minimum(torch.ones_like(gn), cap / gn)
    grad = grad * scale.reshape([B] + [1] * (grad.dim() - 1))
    if os.environ.get("GUIDANCE_STATS"):
        # model-invariant strength: ratio of the applied correction to the model velocity, whether the clip bound,
        # and the metric displacement the correction implies on x0_hat (x0_hat' = x0_hat - sigma*lam*g)
        with torch.no_grad():
            m0 = to_metric(xh.detach()); m1 = to_metric(xh.detach() - float(sigma) * lam * grad)
            STATS.append({"sigma": float(sigma), "lam": float(lam),
                          "ratio": float((lam * grad.reshape(B, -1).norm(dim=1) / vn.clamp(min=1e-12))[0]),
                          "bind": bool(scale[0] < 1.0), "E": float(e),
                          "g_raw": float(gn[0]), "v": float(vn[0]),
                          "d_end": float((m1[0, -1, :2] - m0[0, -1, :2]).norm()),
                          "d_mean": float((m1[0, :, :2] - m0[0, :, :2]).norm(dim=-1).mean())})
    if os.environ.get(debug_tag):
        corr = lam * grad
        lost = float((corr - corr.to(dtype=pred_v.dtype).float()).norm() / corr.norm().clamp(min=1e-12))
        print(f"[gdbg] sigma={sigma:.3f} lam={lam:.4f} E={float(e):.4g} |v|={float(vn[0]):.3f} "
              f"|g_raw|={float(gn[0]):.3f} clip_binds={bool(scale[0] < 1.0)} scale={float(scale[0]):.4f} "
              f"cast_loss={lost:.3e}", flush=True)
    if _trace:
        with torch.no_grad():
            TRACE[-1]["post"] = to_metric(xh.detach() - float(sigma) * lam * grad)[0].cpu().tolist()
    return pred_v + (lam * grad).to(dtype=pred_v.dtype)
