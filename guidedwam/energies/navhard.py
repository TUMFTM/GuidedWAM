"""Differentiable guidance energies for the navhard failure study.

Both energies consume a METRIC ego-frame trajectory `ego` of shape [B, T, 3]
(x fwd, y left, heading) and return a per-batch scalar [B] — the contract of
simwam.guidance.guided_step / drivewam guided_step (through its adapter).

FreespaceFieldEnergy (rules-compliant, camera-derived):
    precomputed BEV distance-to-road field (from 3-camera depth+segmentation)
    sampled bilinearly (differentiable in x, y) + soft clearance to camera-
    detected agent points. Targets DAC + NC.

LaneCenterEnergy (oracle, map-derived):
    pull toward the nearest forward-oriented lane-centerline point with a
    0.5 m dead-zone (same construction as the M1 lane loss). Targets LK + DDC.
"""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter


class FreespaceFieldEnergy:
    def __init__(self, dist_free: np.ndarray, obst_pts: np.ndarray,
                 xmin: float = -10.0, ymin: float = -25.0, cell: float = 0.25,
                 w_dac: float = 1.0, w_nc: float = 1.0, margin: float = 2.0,
                 smooth_sigma_cells: float = 2.0, field_cap: float = 8.0):
        # smooth + cap the C0 distance field: bilinear gradients of the raw
        # field flip direction at cell boundaries and inject per-step jitter
        # through the ~33x/21x denorm amplification (SimWAM); a 0.5 m Gaussian
        # makes the pull direction coherent across denoising steps.
        f = np.minimum(np.asarray(dist_free, np.float32), field_cap)
        if smooth_sigma_cells > 0:
            f = gaussian_filter(f, sigma=smooth_sigma_cells)
        self.field = torch.as_tensor(f, dtype=torch.float32)  # (NX, NY)
        self.obst = torch.as_tensor(obst_pts[:, :2], dtype=torch.float32) if len(obst_pts) else None
        self.xmin, self.ymin, self.cell = xmin, ymin, cell
        self.w_dac, self.w_nc, self.margin = w_dac, w_nc, margin

    def _bilinear(self, xy: torch.Tensor) -> torch.Tensor:
        """xy [..., 2] metric -> field value, differentiable."""
        f = self.field.to(xy.device)
        NX, NY = f.shape
        gx = ((xy[..., 0] - self.xmin) / self.cell).clamp(0, NX - 1.001)
        gy = ((xy[..., 1] - self.ymin) / self.cell).clamp(0, NY - 1.001)
        x0, y0 = gx.floor().long(), gy.floor().long()
        ax, ay = gx - x0.float(), gy - y0.float()
        v = (f[x0, y0] * (1 - ax) * (1 - ay) + f[x0 + 1, y0] * ax * (1 - ay)
             + f[x0, y0 + 1] * (1 - ax) * ay + f[x0 + 1, y0 + 1] * ax * ay)
        return v

    @staticmethod
    def _smooth_traj(xy: torch.Tensor) -> torch.Tensor:
        """[B,T,2] -> 3-tap smoothed copy; chain rule then spreads each
        waypoint's field gradient over its neighbours (coherent pull, no
        per-waypoint tearing)."""
        pad = torch.cat([xy[:, :1], xy, xy[:, -1:]], dim=1)
        return 0.25 * pad[:, :-2] + 0.5 * pad[:, 1:-1] + 0.25 * pad[:, 2:]

    def __call__(self, ego: torch.Tensor) -> torch.Tensor:
        xy = self._smooth_traj(ego[..., :2])
        e = self.w_dac * self._bilinear(xy).mean(dim=-1)
        if self.obst is not None and self.w_nc > 0:
            # soft-min distance (temperature 0.5 m): the hard min switches the
            # active obstacle point between steps and kinks the gradient.
            dall = torch.cdist(xy, self.obst.to(ego.device))  # [B, T, M]
            d = -0.5 * torch.logsumexp(-dall / 0.5, dim=-1)
            e = e + self.w_nc * (torch.relu(self.margin - d) ** 2).mean(dim=-1)
        return e


class LaneCenterEnergy:
    def __init__(self, centerline_pts: np.ndarray, dead_zone: float = 0.5, weight: float = 1.0):
        """centerline_pts (P, >=2): forward-oriented lane-center points, local frame."""
        self.pts = torch.as_tensor(centerline_pts[:, :2], dtype=torch.float32)
        self.dz, self.w = dead_zone, weight

    def __call__(self, ego: torch.Tensor) -> torch.Tensor:
        if len(self.pts) == 0:
            return ego[..., :2].sum(dim=(-2, -1)) * 0.0
        d = torch.cdist(ego[..., :2], self.pts.to(ego.device)).min(dim=-1).values  # [B, T]
        return self.w * (torch.relu(d - self.dz) ** 2).mean(dim=-1)


class TrueDrivableEnergy(FreespaceFieldEnergy):
    """Distance outside the map's drivable-area polygons (0 inside), sampled
    bilinearly on the same BEV grid. Same machinery as the camera freespace
    energy but with an oracle field that has no occlusion holes."""

    def __init__(self, dist_out, weight: float = 1.0, **kw):
        super().__init__(dist_out, np.zeros((0, 2), np.float32),
                         w_dac=weight, w_nc=0.0, **kw)


class BoxClearanceEnergy:
    """Clearance to GT object boxes, forecast at each waypoint's timestamp.

    Signed 2-D box distance (negative inside) keeps a usable gradient even when
    a waypoint lands on top of a box; boxes are advanced with constant velocity
    so the penalty is time-correct; soft-min over boxes avoids the gradient
    kinks a hard nearest-object switch produces.
    """

    def __init__(self, boxes, vel, dt: float = 0.5, margin: float = 1.0,
                 ego_radius: float = 1.0, weight: float = 1.0, temp: float = 0.5,
                 max_range: float = 60.0):
        b = np.asarray(boxes, np.float32).reshape(-1, 7)
        v = np.asarray(vel, np.float32).reshape(-1, 3)
        if len(b):
            keep = np.linalg.norm(b[:, :2], axis=1) < max_range
            b, v = b[keep], v[keep]
        self.n = len(b)
        if self.n:
            self.c = torch.as_tensor(b[:, :2])                 # (N,2) centre
            self.half = torch.as_tensor(b[:, 3:5] * 0.5)       # (N,2) l/2, w/2
            self.yaw = torch.as_tensor(b[:, 6])                # (N,)
            self.v = torch.as_tensor(v[:, :2])                 # (N,2)
        self.dt, self.temp, self.w = dt, temp, weight
        self.margin = margin + ego_radius

    def __call__(self, ego: torch.Tensor) -> torch.Tensor:
        if self.n == 0:
            return ego[..., :2].sum(dim=(-2, -1)) * 0.0
        dev = ego.device
        xy = ego[..., :2]                                       # (B,T,2)
        T = xy.shape[-2]
        t = torch.arange(1, T + 1, device=dev, dtype=xy.dtype) * self.dt      # (T,)
        c = self.c.to(dev)[None, None] + self.v.to(dev)[None, None] * t[None, :, None, None]
        d = xy[..., None, :] - c                                # (B,T,N,2)
        yaw = self.yaw.to(dev)
        cs, sn = torch.cos(-yaw), torch.sin(-yaw)
        local = torch.stack([d[..., 0] * cs + d[..., 1] * (-sn),
                             d[..., 0] * sn + d[..., 1] * cs], dim=-1)
        q = local.abs() - self.half.to(dev)                     # (B,T,N,2)
        outside = torch.linalg.norm(torch.relu(q), dim=-1)
        inside = torch.clamp(torch.maximum(q[..., 0], q[..., 1]), max=0.0)
        sdf = outside + inside                                  # <0 inside the box
        soft = -self.temp * torch.logsumexp(-sdf / self.temp, dim=-1)   # (B,T)
        return self.w * (torch.relu(self.margin - soft) ** 2).mean(dim=-1)
