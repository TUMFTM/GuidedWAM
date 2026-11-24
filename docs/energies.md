# Energies and the camera-only road-edge field

This document gives the exact form of every energy used in the paper, how the guided sampler applies them, and how
the camera-only road-edge field is built. Code references are relative to the repository root.

## 1. Notation

A plan is the model's trajectory of $T = 8$ poses at $\Delta t = 0.5$ s in the current ego frame (rear axle, $x$ forward,
$y$ left):

$$\tau = (p_1, \dots, p_T), \qquad p_t = (x_t, y_t, \psi_t), \qquad p_0 = (0, 0, 0).$$

Every energy maps a plan to a scalar, $E(\tau) \ge 0$, and is differentiable in $\tau$ through bilinear interpolation
of the fields.

**Densified path.** The drivable-area and driving-direction energies are evaluated on the path the scorer simulates,
not just on the waypoints. The current pose $p_0$ is prepended, and $(x, y, \psi)$ is interpolated linearly with
$k = 4$ points per segment:

$$q_m = (1-s)\,p_{t} + s\,p_{t+1}, \qquad s \in \{0, \tfrac14, \tfrac12, \tfrac34\},\ t = 0..T-1, \qquad q_M = p_T,$$

so $M = kT = 32$ and the path has $M + 1 = 33$ poses $q_0, \dots, q_M$, from the current pose $q_0 = p_0$ to the
last waypoint $q_M = p_T$ (`CornerDrivableEnergy._densify`).

**Ego footprint.** The vehicle is the nuPlan Pacifica: length $L = 5.176$ m, width $W = 2.297$ m, rear axle to box
centre $a = 1.461$ m. Its four corners, relative to the rear axle, are

$$c \in \mathcal C = \{(a + \tfrac L2, \pm\tfrac W2),\ (a - \tfrac L2, \pm\tfrac W2)\} = \{(4.049, \pm1.149),\ (-1.127, \pm1.149)\},$$

and at pose $q_m = (x_m, y_m, \psi_m)$ a corner lies at $r_{m,c} = (x_m, y_m) + R(\psi_m)\,c$, with $R$ the 2-D rotation.

**Signed-distance fields.** $\phi(r)$ is a signed distance in metres to the boundary of a region: negative inside,
positive outside, clipped to $[-8, 8]$ m, and sampled bilinearly on a 0.25 m grid. $[u]_+ = \max(u, 0)$.

## 2. Energies

### 2.1 Drivable area (DA): footprint corners

The scorer's drivable-area compliance fails a plan as soon as any footprint corner leaves the drivable area at any
simulated step. The energy penalizes exactly that quantity, with a margin $\delta$ that leaves slack for the
scorer's LQR tracker:

$$E_{\mathrm{DA}}(\tau) = \frac{w}{4(M+1)} \sum_{m=0}^{M} \sum_{c \in \mathcal C} \big[\phi_{\mathrm{DA}}(r_{m,c}) + \delta\big]_+ .$$

Here $\phi_{\mathrm{DA}}(r)$ is the signed distance (m) from the point $r$ to the edge of the drivable area: negative
inside, positive outside, clipped to $\pm 8$ m. A corner more than $\delta$ inside the edge contributes nothing, and
the penalty grows linearly as it approaches or crosses the edge. The normalization $4(M+1) = 132$ averages over the
four corners at each of the 33 poses of the densified path. $\phi_{\mathrm{DA}}$ comes either from the map (oracle) or
from the camera network (learned), as described below.

The penalty is linear in the violation depth, not squared. It depends on heading through $R(\psi_m)$, so the
gradient also turns the vehicle, not only moves it. The code is `CornerDrivableEnergy` (oracle field) and
`LearnedCornerDrivableEnergy` (camera field) in `guidedwam/energies/navhard_fp.py`.

**Oracle field** (privileged, Tables 2-3, Figures 1 and 3):
- **Region.** The union of the map layers the scorer's `PDMDrivableMap` uses: ROADBLOCK, ROADBLOCK_CONNECTOR,
  INTERSECTION, CARPARK_AREA.
- **Grid.** The union is rasterized at the cell centres of $x \in [-10, 60)$ m, $y \in [-25, 25)$ m, a $280 \times 200$
  grid (`scripts/fields/precompute_oracle_fields.py`, stored as the unsigned distance outside, `ora_<token>.npz`).
- **Signed distance.** The energy forms $\phi = d_{\mathrm{out}} - d_{\mathrm{in}}$ from two Euclidean distance
  transforms, clips it to $\pm 8$ m and smooths it with a Gaussian of $\sigma = 1$ cell (0.25 m).
- **Margin.** $\delta = 0.5$ m.

**Learned field** (camera only, Tables 1, 4, 5 and Figure 2):
- **Field.** The output $\hat\phi$ of the road-edge network of Section 4, on its own grid: $x \in [0, 32]$ m,
  $y \in [-32, 32]$ m, 0.25 m cells, $128 \times 256$. It is clipped to $\pm 8$ m and not smoothed.
- **Outside the window.** Corners outside the grid are unknown. They get no penalty and no gradient, instead of
  having the border value smeared outward.
- **Margins.** $\delta = 1.0$ m (DriveWAM) and $1.5$ m (SimWAM, DriveLaW). The predicted edge tends to lie inside the
  true one, so a wider band buys recall (Section 4.5).

### 2.2 Lane keeping (LK)

Each waypoint is attracted to the nearest forward-oriented lane centreline point, with a dead zone $d_0 = 0.5$ m.
Let $\mathbf{u}_t = (x_t, y_t)^\top \in \mathbb{R}^2$ be the position of waypoint $t$ and $\ell_1, \dots, \ell_J \in
\mathbb{R}^2$ the positions of the lane centreline points, both in the ego frame. Then
$d_t = \min_j \|\mathbf{u}_t - \ell_j\|_2$ is the Euclidean distance from waypoint $t$ to the nearest lane point, and

$$E_{\mathrm{LK}}(\tau) = \frac{w}{T} \sum_{t=1}^{T} \big[d_t - d_0\big]_+^2, \qquad d_t = \min_{j=1..J} \big\|\mathbf{u}_t - \ell_j\big\|_2 .$$

$d_t$ is the distance to the nearest *sampled* centreline point, not the perpendicular distance to the centreline
polyline. The points are 0.5 m apart along each lane (median and 90th percentile of the spacing), so $d_t$ exceeds the
perpendicular distance $d_\perp$ by at most $\sqrt{d_\perp^2 + 0.25^2} - d_\perp$: 0.25 m on the centreline, inside
the dead zone, and 0.03 m at $d_\perp = 1$ m. A waypoint within $d_0$ of a lane centre contributes nothing. Beyond that, the penalty grows quadratically with the
distance, and its gradient points from the waypoint toward its nearest lane point.

The centreline points $\ell_j$ (`scripts/fields/precompute_lane_fields.py`, `lanes_<token>.npz`) are every second
point of the discretized baseline paths of the LANE and LANE_CONNECTOR objects that intersect the 120 m x 120 m square centred on the ego (nuPlan's `get_proximal_map_objects` with radius 60 m). Only points
that satisfy all three conditions below are kept:
- $-10 < x < 70$ m;
- $|y| < 30$ m;
- lane heading within $\pm 80°$ of the ego heading.

The heading filter keeps oncoming lanes out of the minimum, so the energy also pulls plans back from the wrong
direction. The code is `LaneCenterEnergy` in `guidedwam/energies/navhard.py`. The field is privileged (map).

### 2.3 Driving direction (DD)

**What the scorer measures.** The driving-direction compliance of the PDM scorer
(`pdm_scorer.py`, `_calculate_driving_direction_compliance`) does not compare headings. It is a test on the lanes:
- **On-route lanes.** The route is a sequence of roadblocks. Its lanes are the LANE and LANE_CONNECTOR polygons of
  those roadblocks (`route_lane_ids` in the metric cache).
- **Why lanes stand for direction.** nuPlan lanes are directed, and a roadblock groups lanes of one travel direction
  only. The lanes of the opposite carriageway therefore belong to other roadblocks, which are not on the route.
- **The test.** At every simulated pose, the box centre counts as being "in oncoming traffic" if it lies in no
  on-route lane. The scorer sums the distance the centre travels in that state, sets it to zero inside INTERSECTION
  polygons (where connectors of all directions overlap), and takes the maximum of this sum over any 1 s window.
- **Score.** Below 2 m it gives 1, below 6 m 0.5, otherwise 0.

The quantity is thus "distance driven outside the route's own lanes". This is the oncoming lane in practice, but also
any other off-route lane, such as a wrong turn. A car on a correct lane but pointing backwards would pass.

**The energy.** It replaces the scorer's 0/1 "outside" test by the depth outside, which has a gradient. The test is
evaluated at the box centre of the ego at each pose $q_m$ of the densified path (Section 1):
$o_m = (x_m + a\cos\psi_m,\ y_m + a\sin\psi_m)$. Here $m$ indexes the 33 poses of the one ego box along the plan, not
different boxes. Then

$$E_{\mathrm{DD}}(\tau) = w \sum_{m=1}^{M} \underbrace{\|o_m - o_{m-1}\|}_{\text{step length}}\,\big[\phi_{\mathrm{DD}}(o_m) + \delta_r\big]_+, \qquad \delta_r = 0,$$

a discretized path integral $\int [\phi_{\mathrm{DD}}]_+\,\mathrm ds$ of the off-route depth along the centre's path.
A step counts only when its end point is off-route, as in the scorer, and it counts in proportion to how far
off-route it is. Two differences from the scorer:
- the energy sums over the whole 4 s horizon instead of the worst 1 s window;
- its unit is m² (metres travelled × metres off-route) instead of metres.

**The field.** $\phi_{\mathrm{DD}}$ is the signed distance to the union of the on-route lane polygons. Route and
polygons are both read from the metric cache the scorer itself uses (`scripts/fields/precompute_route_fields.py`,
`route_<token>.npz`, same grid as the oracle DA field). It is smoothed with $\sigma = 1$ cell and set to $-8$ m inside
INTERSECTION polygons, so that no energy accrues where the scorer does not count. The code is `OffRouteEnergy` in `guidedwam/energies/navhard_fp.py`. The field is privileged (map and
route).

### 2.4 Speed cap

The plan's speed is the finite difference between consecutive waypoints, with the current pose as predecessor of
the first:

$$v_t = \frac{\sqrt{\|(x_t, y_t) - (x_{t-1}, y_{t-1})\|^2 + 10^{-9}}}{\Delta t}, \qquad t = 1..T,$$

and the energy is a squared hinge above a cap $v_{\max}$ given at inference:

$$E_{v}(\tau) = w \sum_{t=1}^{T} \big[v_t - v_{\max}\big]_+^2 .$$

The cap is a single number, so the energy needs no map or perception input. The code is `SpeedCapEnergy` in
`guidedwam/energies/kinematic.py`. It is enabled for any arm through `SPEED_CAP` (m/s) and `SPEED_CAP_W` ($w$), or
alone through `ARM=speedcap`.

**Preconditioning for absolute-pose models.** $v_t$ depends on the difference of two consecutive positions. When a
model denoises absolute poses (SimWAM, DriveLaW), the gradient of $E_v$ therefore couples neighbouring waypoints: its
Hessian is a second-difference operator over time. The plain Euler step then mostly moves the last waypoint, overshoots
and alternates, and the plans zig-zag. DriveWAM denoises pose differences and is not affected. For the absolute-pose
models the metric gradient $g = \partial E_v / \partial \tau$ is preconditioned before it is pulled back into the
latent:

$$\tilde g_t = \sum_{s=1}^{t} \sum_{r=s}^{T} g_r, \qquad \text{i.e. } \tilde g = C C^\top g, \quad C_{ts} = \mathbb 1[s \le t].$$

$C$ maps the segment displacements $d_s = p_s - p_{s-1}$ to the positions $p_t = \sum_{s \le t} d_s$, so $\tilde g$ is
the position change of a descent step in the displacements: $\tilde g = C\,\partial E_v/\partial d$. For an energy of
the displacements alone, the preconditioned Hessian is that of the displacements, which is diagonal. The step then
slows all segments of a plan together. The preconditioner applies to the speed cap only (`SPEED_CAP_PRECOND=1`;
`_precondition_cumsum` in `guidedwam/sampler/guidance.py`). The field energies are unchanged.

### 2.5 Compositions

Composed arms add weighted energies, $E = \sum_i w_i E_i$, under a single strength $\lambda$. The weights are
relative to the first energy:

| arm (`ARM`) | energies | weights used in Table 3 |
|---|---|---|
| `corner` | DA (oracle) | $w_{\mathrm{DA}} = 1$ |
| `map` | LK | $w_{\mathrm{LK}} = 1$ |
| `offroute` | DD | $w_{\mathrm{DD}} = 1$ |
| `mapcorner` | LK + DA | $w_{\mathrm{LK}} = 1$; $w_{\mathrm{DA}} = 1$ (DriveLaW 2) |
| `corneroffroute` | DA + DD | $w_{\mathrm{DA}} = 1$, $w_{\mathrm{DD}} = 0.06$ |
| `mapcorneroffroute` | LK + DA + DD | $w_{\mathrm{LK}} = 1$; $w_{\mathrm{DA}} = 1$ (DriveLaW 2); $w_{\mathrm{DD}} = 0.12$ (DriveWAM), 0.15 (SimWAM, DriveLaW) |
| `cornerlearn` | DA (learned) | $w_{\mathrm{DA}} = 1$ |

The run variables are `W_DAC` (first energy), `W_NC` (second energy) and `CORNER_W` (DA in the three-term arm).
$E_{\mathrm{DD}}$ is a sum over path length, not a mean, and is about 100 times larger than $E_{\mathrm{DA}}$, hence the
small DD weights.

## 3. The guided sampler

All three models are flow-matching samplers with $x_\sigma = (1-\sigma)\,x_0 + \sigma\,\varepsilon$ and velocity target
$v = \varepsilon - x_0$, integrated by Euler steps from $\sigma = 1$ to $0$. At each step $k$ with model velocity $\hat
v_k$ (`guidedwam/sampler/guidance.py`):

1. **One-step clean estimate, Jacobian-free.** $\hat x_0 = x_k - \sigma_k\,\mathrm{sg}(\hat v_k)$, where
   $\mathrm{sg}$ stops the gradient, so nothing is backpropagated through the denoiser.
2. **Energy gradient in latent space.** $g = \nabla_{\hat x_0}\, E\big(f(\hat x_0)\big)$, with $f$ the model's
   differentiable map from action latents to the metric plan $\tau$:
   - **DriveWAM:** de-normalize the $[-1,1]$ quantile-normalized pose differences, then a cumulative sum
     (`guidedwam/sampler/drivewam.py`).
   - **SimWAM and DriveLaW:** an affine de-normalization of absolute poses
     ($x = 33.37(u+1) - 1.57$, $y = 21(u+1) - 19.68$, $\psi = 1.765(u+1) - 1.67$; `guidedwam/sampler/simwam.py`).
3. **Velocity bound.** Per sample, $g \leftarrow g \cdot \min\!\big(1,\ c\,\|\hat v_k\| / (\lambda(\sigma_k)\,\|g\|)\big)$
   with $c = 12$. This keeps $\|\lambda g\| \le c\,\|\hat v_k\|$, using the velocity of the same step.
4. **Corrected velocity and Euler step.** $\tilde v_k = \hat v_k + \lambda(\sigma_k)\, g$, then
   $x_{k+1} = x_k + \tilde v_k\,(\sigma_{k+1} - \sigma_k)$. Since $\sigma_{k+1} < \sigma_k$, the step moves both
   $x_{k+1}$ and the implied $\hat x_0$ along $-\nabla E$.

**Window.** $\lambda(\sigma)$ is a trapezoid of height $\lambda$:

$$\lambda(\sigma) = \lambda \cdot \begin{cases} 0 & \sigma \ge 0.95 \ \text{or}\ \sigma \le 0.05 \\
(0.95 - \sigma)/0.15 & 0.8 < \sigma < 0.95 \\ 1 & 0.3 \le \sigma \le 0.8 \\ (\sigma - 0.05)/0.25 & 0.05 < \sigma < 0.3 \end{cases}$$

**Settings.**
- Sampler steps: DriveWAM 10 action steps (after 3 video steps), SimWAM 20, DriveLaW 10.
- Strengths of the paper: $\lambda = 4$ (DriveWAM), $0.05$ (SimWAM, DriveLaW) on the learned field.
- **Options not used at the operating points.**
  - `normalize`: divides each energy's gradient by its own per-sample norm before summing, instead of the fixed
    weights. It is used only in the stabilization ablation of the paper, where it scores the same as fixed weights
    on the full split (`NORM=1` for SimWAM, `GUIDANCE_NORMALIZE=1` for DriveWAM).
  - The Jacobian form replaces $g$ by the gradient through the network, i.e. without the stop-gradient in step 1
    (`JAC=1`, Table 5 only).

**Role of the stabilizers.** The window carries the stability at the operating points: a uniform weight scores below
the stock model on DriveWAM. The velocity bound is a safeguard. It is not active at any strength of the strength study
and limits the correction only far above the operating points.

## 4. Camera-only road-edge field

The network predicts $\phi_{\mathrm{DA}}$ (Section 2.1) from the cameras of the current frame only. It sees no map,
no LiDAR and no route at inference. Its training targets are the same signed distance as the oracle field.

### 4.1 Architecture

- **Base.** NAVSIM's Latent TransFuser (LTF, `navsim/agents/transfuser`, `TransfuserConfig.latent = True`): a
  ResNet-34 image encoder whose LiDAR branch is replaced by a learned latent BEV input, fused by TransFuser's
  transformer blocks, with TransFuser's BEV semantic decoder.
- **Change.** Only the last $1 \times 1$ convolution of the BEV head changes, from 7 classes to 1 regression channel.
  The output is a $128 \times 256$ map at 0.25 m: $x \in [0, 32]$ m forward, $y \in [-32, 32]$ m (TransFuser's BEV
  frame).
- **Initialization.** All other weights are warm-started from the released `ltf_seed_0.ckpt` (navsim_baselines); the
  new $1 \times 1$ convolution is trained from scratch.
- **Inputs.**
  - Camera: the left, front and right cameras (`cam_l0`, `cam_f0`, `cam_r0`) of the current frame. Rows 28..H−28 are
    kept from each, and columns 416..W−416 from the side cameras. The three are stitched side by side and resized to
    $1024 \times 256$, exactly TransFuser's camera feature.
  - Status vector of 8 values: the one-hot driving command (4), the ego velocity (2) and the ego acceleration (2).
- **Code.** `scripts/fields/bevsdf_train.py`, `build_model`.

### 4.2 Targets

For every training scene (`scripts/fields/bevsdf_targets.py`):
1. Take the polygons of ROADBLOCK, ROADBLOCK_CONNECTOR, INTERSECTION and CARPARK_AREA that intersect the 140 m x 140 m square centred on the ego (radius 70 m), which covers the whole output grid. This
   is the scorer's drivable area, not the LANE + INTERSECTION label of TransFuser's stock semantic head.
2. Test containment at the cell centres of the output grid.
3. Set $\phi = d_{\mathrm{out}} - d_{\mathrm{in}}$ from two Euclidean distance transforms (metres), clipped to $\pm 8$ m.

The reason for regressing a distance instead of a class: measured on navhard, even a perfect prediction of the
stock head's LANE + INTERSECTION class detects only 79.8 % of the footprint corners that violate the scorer's drivable
area. The released checkpoint detects 16.9 %. The loss of a class label does not reward boundary precision, and a
distance target does.

### 4.3 Loss and training

- **Loss.** Smooth-L1 ($\beta = 0.5$ m), weighted toward the edge, since the energy only reads the field near it:

  $$\mathcal L = \frac{1}{|\Omega|}\sum_{(i,j)\in\Omega} \big(1 + 4\,e^{-|\phi_{ij}|/1\,\mathrm m}\big)\ \mathrm{SmoothL1}_{0.5}\big(\hat\phi_{ij} - \phi_{ij}\big).$$

- **Data.** navtrain, 103,288 frames from 1,192 logs. The current frame of every navtrain scene is used.
  Validation is 5 % of the logs, split by log so that near-duplicate frames of one log never straddle the split.
- **Optimization.**
  - AdamW, learning rate $10^{-4}$, weight decay $10^{-4}$, one-cycle schedule with 5 % warm-up.
  - 20 epochs, batch 32, gradient-norm clip 1.0, fp32.
  - About 12 min per epoch on one RTX 4090, with the images pre-decoded once (`bevsdf_cache_images.py`).
- **Checkpoint.** The epoch with the lowest validation MAE within 1 m of the edge is kept, which is epoch 20.

| validation (navtrain logs held out) | value |
|---|---|
| MAE of $\hat\phi$, all cells | 0.32 m |
| MAE within 1 m of the edge | 0.40 m |
| sign accuracy (inside / outside) | 97.6 % |
| IoU of the predicted drivable area | 94.5 % |

### 4.4 Inference

- **Frequency.** The field is computed once per scene, from the current frame, before sampling:
  `scripts/fields/bevsdf_eval.py` with `SAVE_FIELDS`, giving `bsdf_<token>.npz` (float16).
- **Use during sampling.** At every guided step the sampler reads it by bilinear interpolation; the network is not
  re-run per step.
- **Cost.** 56.0 M parameters. On one RTX 4090, batch 1, fp32, median over 100 navhard frames: 10.4 ms for the
  forward pass, 12.0 ms including the camera crop, stitch and resize; peak GPU memory 0.36 GB. Once per scene, this is
  negligible next to the WAM sampler (e.g. SimWAM: 20 steps of about 16 ms).

### 4.5 Quality on navhard

Measured on 1,199 navhard scenes where the stock DriveWAM plan fails drivable-area compliance (1,185 of them
synthetic stage-two scenes), against the map-derived target of Section 4.2:

| subset | MAE | MAE within 1 m of edge | sign accuracy | IoU |
|---|---|---|---|---|
| all | 0.91 m | 1.20 m | 91.9 % | 76.1 % |
| stage one (real images) | 0.45 m | 0.72 m | 96.9 % | 89.8 % |
| stage two (3DGS renderings) | 0.92 m | 1.20 m | 91.8 % | 75.9 % |

The field is much better on real images than on the rendered stage-two images it never saw in training. The
corner test below counts how many of the plan's footprint corners that violate the scorer's drivable area the
energy sees, i.e. $\hat\phi + \delta > 0$. It also counts how many compliant corners it flags falsely. The test uses
16,407 violating and 70,677 compliant corners.

| field | margin $\delta$ | violating corners detected | compliant corners flagged |
|---|---|---|---|
| oracle (map) | 0 | 100.0 % | 0.0 % |
| released LTF semantic head | – | 16.9 % | 1.3 % |
| this network | 0.0 m | 23.0 % | 1.2 % |
| this network | 0.5 m | 42.2 % | 5.1 % |
| this network | 1.0 m | 62.2 % | 12.2 % |
| this network | 1.5 m | 75.4 % | 22.0 % |

Detection grows with the depth of the violation. At $\delta = 0$ it is 8.8 % for corners 0.25-0.5 m outside and
74.6 % beyond 2 m. Errors grow with range: the MAE within 1 m of the edge is 1.03 m at 0-8 m and 1.31 m at 16-32 m.

### 4.6 Reproducing

`scripts/fields/build_fields.sh learned` runs the whole chain: targets, index, image cache, training, inference
over all 5,912 navhard scenes. It needs the navtrain logs, the trainval sensor blobs and `ltf_seed_0.ckpt`
(`docs/installation.md`).

## 5. Post-hoc projection baseline

The projection baseline of Table 2 acts on the finished stock plan $\tau_0$ and uses the same footprint-corner
constraint as the oracle drivable-area energy, with $\delta = 0.5$ m (`scripts/ablation/project_navhard.py`). It
minimizes, over a correction $\Delta$ on position and heading,

$$\mu\, E_{\mathrm{DA}}(\tau_0 + \Delta) + \|\Delta\|_2^2 + \eta\, S(\tau_0 + \Delta),$$

- **Optimizer.** Adam, learning rate 0.01, at most 1,000 iterations.
- **Penalty weight.** $\mu$ escalates through 1, 10, 100 and 1,000 whenever the violation stops decreasing.
- **Stop.** When every corner of the future poses satisfies $\phi_{\mathrm{DA}} + \delta \le 0$.
- **Pass-through.** Plans without a violation are unchanged. Plans that still violate at the last iteration keep the
  stock trajectory and are counted as failed.
- **Smoothness term.** $S$ is the sum of squared second differences of the waypoints, with the current pose
  prepended. The projector of Table 2 uses $\eta = 0$. The smoothed variant sets $\eta$ through `PROJ_SMOOTH`
  (0.1, 1 and 10 in the paper's supplementary material).
