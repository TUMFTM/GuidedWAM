"""Footprint-aware drivable-area energy.

The plain TrueDrivableEnergy penalises only the 8 waypoint CENTRES, but navsim's
DAC is evaluated on the ego FOOTPRINT swept along a densely interpolated,
kinematically simulated 4 s path. A plan whose centres all sit inside the
polygon can still clip the boundary with its corners, or exit between waypoints
— so the naive energy stays silent on most scenes that actually fail DAC.

Two corrections, both mirroring the metric:
  * erode the drivable region by the ego half-width before the distance
    transform, so "within half a car of the edge" already counts as violating;
  * prepend the current ego pose and linearly densify the waypoint polyline
    before sampling, so between-waypoint excursions are seen.
"""
import os
import zlib

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


class FootprintDrivableEnergy:
    def __init__(self, dist_out: np.ndarray, weight: float = 1.0,
                 ego_half_width: float = 1.0, densify: int = 4, cell: float = 0.25,
                 xmin: float = -10.0, ymin: float = -25.0,
                 smooth_sigma_cells: float = 2.0, field_cap: float = 8.0):
        inside = np.asarray(dist_out) <= 1e-6
        r = max(int(round(ego_half_width / cell)), 1)
        eroded = binary_erosion(inside, _disk(r)) if inside.any() else inside
        field = (distance_transform_edt(~eroded) * cell).astype(np.float32) if eroded.any() \
            else np.full(inside.shape, 50.0, np.float32)
        field = np.minimum(field, field_cap)
        if smooth_sigma_cells > 0:
            field = gaussian_filter(field, sigma=smooth_sigma_cells)
        self.field = torch.as_tensor(field, dtype=torch.float32)
        self.xmin, self.ymin, self.cell, self.w, self.k = xmin, ymin, cell, weight, densify

    @staticmethod
    def _densify(xy: torch.Tensor, k: int) -> torch.Tensor:
        """Prepend the current pose (origin) and linearly interpolate k-fold."""
        B = xy.shape[0]
        pts = torch.cat([xy.new_zeros(B, 1, 2), xy], dim=1)          # (B, T+1, 2)
        a, b = pts[:, :-1], pts[:, 1:]
        ts = torch.linspace(0, 1, k + 1, device=xy.device, dtype=xy.dtype)[:-1].view(1, 1, -1, 1)
        seg = a[:, :, None, :] * (1 - ts) + b[:, :, None, :] * ts    # (B, T, k, 2)
        return torch.cat([seg.flatten(1, 2), pts[:, -1:]], dim=1)

    def _bilinear(self, xy: torch.Tensor) -> torch.Tensor:
        f = self.field.to(xy.device)
        NX, NY = f.shape
        gx = ((xy[..., 0] - self.xmin) / self.cell).clamp(0, NX - 1.001)
        gy = ((xy[..., 1] - self.ymin) / self.cell).clamp(0, NY - 1.001)
        x0, y0 = gx.floor().long(), gy.floor().long()
        ax, ay = gx - x0.float(), gy - y0.float()
        return (f[x0, y0] * (1 - ax) * (1 - ay) + f[x0 + 1, y0] * ax * (1 - ay)
                + f[x0, y0 + 1] * (1 - ax) * ay + f[x0 + 1, y0 + 1] * ax * ay)

    def __call__(self, ego: torch.Tensor) -> torch.Tensor:
        return self.w * self._bilinear(self._densify(ego[..., :2], self.k)).mean(dim=-1)


class CornerDrivableEnergy:
    """Footprint-corner drivable energy — measures what the PDM scorer measures.

    The scorer zeroes DAC if ANY of the four Pacifica footprint corners leaves the
    drivable polygons at ANY simulated step.  ``FootprintDrivableEnergy`` samples
    the rear-axle point on a disk-eroded field (width only); the front corners are
    4.05 m ahead of that point and sweep outside on turns while the axle stays in.
    This energy places the actual rectangle at every densified pose (x, y, heading)
    and samples a SIGNED distance-to-edge field (>0 outside, <0 inside) at its four
    corners.  ``margin`` (m) penalises corners closer than that to the edge so the
    LQR tracker's deviation from the plan has slack.  Differentiable in heading.
    """
    L, W, RA2C = 5.176, 2.297, 1.461                       # nuplan Pacifica

    def __init__(self, dist_out: np.ndarray, weight: float = 1.0, margin: float = 0.3,
                 densify: int = 4, cell: float = 0.25, xmin: float = -10.0, ymin: float = -25.0,
                 smooth_sigma_cells: float = 1.0, field_cap: float = 8.0):
        inside = np.asarray(dist_out) <= 1e-6
        d_out = (distance_transform_edt(~inside) * cell) if (~inside).any() else np.zeros(inside.shape)
        d_in = (distance_transform_edt(inside) * cell) if inside.any() else np.zeros(inside.shape)
        signed = np.clip(d_out - d_in, -field_cap, field_cap).astype(np.float32)
        if not inside.any():
            signed = np.full(inside.shape, field_cap, np.float32)
        if smooth_sigma_cells > 0:
            signed = gaussian_filter(signed, sigma=smooth_sigma_cells)
        self.field = torch.as_tensor(signed, dtype=torch.float32)
        hx, hy = self.L / 2, self.W / 2
        self.corners = torch.tensor([[self.RA2C + hx, hy], [self.RA2C + hx, -hy],
                                     [self.RA2C - hx, hy], [self.RA2C - hx, -hy]], dtype=torch.float32)
        self.xmin, self.ymin, self.cell, self.w, self.k, self.margin = xmin, ymin, cell, weight, densify, margin

    @staticmethod
    def _densify(p: torch.Tensor, k: int) -> torch.Tensor:
        """Prepend the current pose (origin, heading 0) and linearly interpolate k-fold (x, y, heading)."""
        B = p.shape[0]
        pts = torch.cat([p.new_zeros(B, 1, 3), p], dim=1)
        a, b = pts[:, :-1], pts[:, 1:]
        ts = torch.linspace(0, 1, k + 1, device=p.device, dtype=p.dtype)[:-1].view(1, 1, -1, 1)
        seg = a[:, :, None, :] * (1 - ts) + b[:, :, None, :] * ts
        return torch.cat([seg.flatten(1, 2), pts[:, -1:]], dim=1)

    def _bilinear(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        f = self.field.to(x.device)
        NX, NY = f.shape
        gx = ((x - self.xmin) / self.cell).clamp(0, NX - 1.001)
        gy = ((y - self.ymin) / self.cell).clamp(0, NY - 1.001)
        x0, y0 = gx.floor().long(), gy.floor().long()
        ax, ay = gx - x0.float(), gy - y0.float()
        return (f[x0, y0] * (1 - ax) * (1 - ay) + f[x0 + 1, y0] * ax * (1 - ay)
                + f[x0, y0 + 1] * (1 - ax) * ay + f[x0 + 1, y0 + 1] * ax * ay)

    def corner_xy(self, ego: torch.Tensor):
        p = self._densify(ego[..., :3], self.k)                       # (B, M, 3)
        c, s = torch.cos(p[..., 2:3]), torch.sin(p[..., 2:3])          # (B, M, 1)
        cr = self.corners.to(ego.device)
        cx, cy = cr[:, 0].view(1, 1, 4), cr[:, 1].view(1, 1, 4)
        X = p[..., 0:1] + c * cx - s * cy
        Y = p[..., 1:2] + s * cx + c * cy
        return X, Y                                                    # (B, M, 4)

    def __call__(self, ego: torch.Tensor) -> torch.Tensor:
        X, Y = self.corner_xy(ego)
        pen = torch.relu(self._bilinear(X, Y) + self.margin)           # >0 when outside or within margin
        return self.w * pen.mean(dim=(-2, -1))


def perturb_field(sdf: np.ndarray) -> np.ndarray:
    """Field-error sensitivity (off unless set): FIELD_SHIFT [m] moves the predicted road edge, > 0 inward (the
    drivable area shrinks), < 0 outward; FIELD_NOISE [m] adds zero-mean Gaussian noise smoothed over
    FIELD_NOISE_CELLS cells (default 4 = 1 m) and rescaled to that standard deviation, seeded per scene."""
    shift, noise = float(os.environ.get("FIELD_SHIFT", "0")), float(os.environ.get("FIELD_NOISE", "0"))
    if shift:
        sdf = sdf + shift
    if noise:
        rng = np.random.default_rng(zlib.crc32(sdf.tobytes()))
        n = gaussian_filter(rng.standard_normal(sdf.shape).astype(np.float32),
                            sigma=float(os.environ.get("FIELD_NOISE_CELLS", "4")))
        sdf = sdf + noise * n / max(float(n.std()), 1e-12)
    return sdf


class LearnedCornerDrivableEnergy(CornerDrivableEnergy):
    """CornerDrivableEnergy over a LEARNED signed-distance field (bevsdf_train.py output).

    Differences from the oracle version:
      * the field is already SIGNED (>0 outside the drivable area, <0 inside), so no distance
        transform is applied to it;
      * its grid is the TransFuser BEV frame — field[i, j], i = x/0.25 over x in [0, 32] m forward,
        j = y/0.25 + 128 over y in [-32, 32] m — not the [-10, 60] x [-25, 25] oracle grid;
      * everything OUTSIDE that window is UNKNOWN, not drivable and not obstacle. The oracle field
        covers 60 m ahead, this one 32 m, and 7% of violating corners on navhard fall beyond it.
        Clamping to the border (what bilinear sampling does by default) would smear the edge value
        outwards and invent a barrier — or invent free space — wherever the net cannot see. Corners
        outside the window are therefore given sd = -inf, i.e. no penalty and no gradient.

    Expects the raw net output; pass `smooth_sigma_cells` > 0 to blur it, which trades boundary
    sharpness for gradient smoothness.
    """

    def __init__(self, sdf: np.ndarray, weight: float = 1.0, margin: float = 0.3,
                 densify: int = 4, smooth_sigma_cells: float = 0.0, field_cap: float = 8.0):
        f = perturb_field(np.asarray(sdf, dtype=np.float32))
        f = np.clip(f, -field_cap, field_cap)
        if smooth_sigma_cells > 0:
            f = gaussian_filter(f, sigma=smooth_sigma_cells)
        self.field = torch.as_tensor(f, dtype=torch.float32)
        hx, hy = self.L / 2, self.W / 2
        self.corners = torch.tensor([[self.RA2C + hx, hy], [self.RA2C + hx, -hy],
                                     [self.RA2C - hx, hy], [self.RA2C - hx, -hy]], dtype=torch.float32)
        # BEV frame of TransfuserConfig: cell i spans x in [i*0.25, (i+1)*0.25), j spans y from -32 m
        self.cell, self.xmin, self.ymin = 0.25, 0.0, -32.0
        self.w, self.k, self.margin = weight, densify, margin

    def _in_window(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        NX, NY = self.field.shape
        return ((x >= self.xmin) & (x <= self.xmin + (NX - 1) * self.cell)
                & (y >= self.ymin) & (y <= self.ymin + (NY - 1) * self.cell))

    def __call__(self, ego: torch.Tensor) -> torch.Tensor:
        X, Y = self.corner_xy(ego)
        sd = self._bilinear(X, Y)
        pen = torch.relu(sd + self.margin)
        pen = torch.where(self._in_window(X, Y), pen, torch.zeros_like(pen))   # unknown -> no force
        return self.w * pen.mean(dim=(-2, -1))


class OffRouteEnergy:
    """Energy for the DRIVING DIRECTION metric — the second gate after drivable area.

    pdm_scorer defines "oncoming traffic" as: the ego CENTRE is not inside any drivable polygon that
    is ON-ROUTE (pdm_scorer.py:359-361). It then sums the distance travelled in that state over a
    rolling 1 s window, ignoring progress inside INTERSECTION polygons, and thresholds at 2 m (score
    1.0 -> 0.5) and 6 m (-> 0.0).

    So this penalises DISTANCE TRAVELLED off-route, not mere presence: the integrand is
    step_length * relu(signed_distance_off_route + margin), suppressed inside intersections exactly
    where the metric suppresses it. One point (the centre), not four corners — the metric uses the
    centre only.

    Fields from precompute_route_fields.py: dist_off (m outside the on-route union, 0 inside) and
    in_inter. Validated against scored outcomes on DriveLaW: compliance-zero scenes accumulate a
    median 19.9 m off-route, compliance-one scenes 0.00 m.
    """
    RA2C = 1.461                                           # rear axle -> bbox centre

    def __init__(self, dist_off: np.ndarray, in_inter: np.ndarray, weight: float = 1.0,
                 margin: float = 0.0, densify: int = 4, cell: float = 0.25,
                 xmin: float = -10.0, ymin: float = -25.0, smooth_sigma_cells: float = 1.0,
                 field_cap: float = 8.0):
        inside = np.asarray(dist_off) <= 1e-6
        d_out = (distance_transform_edt(~inside) * cell) if (~inside).any() else np.zeros(inside.shape)
        d_in = (distance_transform_edt(inside) * cell) if inside.any() else np.zeros(inside.shape)
        signed = np.clip(d_out - d_in, -field_cap, field_cap).astype(np.float32)
        if not inside.any():
            signed = np.full(inside.shape, field_cap, np.float32)
        if smooth_sigma_cells > 0:
            signed = gaussian_filter(signed, sigma=smooth_sigma_cells)
        # the metric ignores off-route progress inside intersections: make the field free there
        signed = np.where(np.asarray(in_inter, bool), -field_cap, signed)
        self.field = torch.as_tensor(signed, dtype=torch.float32)
        self.cell, self.xmin, self.ymin = cell, xmin, ymin
        self.w, self.k, self.margin = weight, densify, margin

    _densify = staticmethod(CornerDrivableEnergy._densify)   # staticmethod, else self is passed too
    _bilinear = CornerDrivableEnergy._bilinear

    def __call__(self, ego: torch.Tensor) -> torch.Tensor:
        p = self._densify(ego[..., :3], self.k)                       # (B, M, 3)
        cx = p[..., 0] + self.RA2C * torch.cos(p[..., 2])
        cy = p[..., 1] + self.RA2C * torch.sin(p[..., 2])
        pen = torch.relu(self._bilinear(cx, cy) + self.margin)        # >0 when off-route
        step = torch.zeros_like(cx)
        step[..., 1:] = torch.sqrt((cx[..., 1:] - cx[..., :-1]) ** 2
                                   + (cy[..., 1:] - cy[..., :-1]) ** 2 + 1e-12)
        return self.w * (step * pen).sum(dim=-1)                      # metres-off-route, softened
