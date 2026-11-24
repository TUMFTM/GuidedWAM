"""Train a camera-only BEV signed-distance field for the corner energy.

Architecture: navsim's Latent TransFuser (camera-only, no LiDAR branch), warm-started from
`ltf_seed_0.ckpt`, with the 7-class BEV semantic head replaced by a 1-channel regression head.

Target: signed distance to the boundary of the scorer's DAC drivable area (ROADBLOCK +
ROADBLOCK_CONNECTOR + INTERSECTION + CARPARK_AREA), negative inside, clipped to +-8 m, on the
128 x 256 @ 0.25 m BEV grid (bevsdf_targets.py).

Why regression and not the stock 7-class head: measured on navhard, a *perfect* prediction of the
stock head's LANE+INTERSECTION label detects 79.8% of DAC-violating footprint corners while the
released checkpoint detects 16.9%. The gap is boundary precision, which a class label does not
reward and a distance does.

Loss: smooth-L1 on the signed distance, weighted towards the boundary, since only the first metre
either side of the edge matters for the energy.

Env: EPOCHS BATCH LR WORKERS VAL_FRAC OUT_DIR RESUME
"""
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_model import TransfuserModel
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from guidedwam import paths

EXP = paths.EXP
SENSOR_ROOT = paths.NAVSIM_DATA / "sensor_blobs/trainval"
TARGETS = EXP / "bevsdf_navtrain"
CKPT_INIT = str(paths.WEIGHTS / "ltf_seed_0.ckpt")
OUT = Path(os.environ.get("OUT_DIR", EXP / "bevsdf_run1"))
OUT.mkdir(parents=True, exist_ok=True)

EPOCHS = int(os.environ.get("EPOCHS", "20"))
BATCH = int(os.environ.get("BATCH", "32"))
LR = float(os.environ.get("LR", "1e-4"))
WORKERS = int(os.environ.get("WORKERS", "10"))
VAL_FRAC = float(os.environ.get("VAL_FRAC", "0.05"))
CAP = 8.0
BW = 1.0            # boundary weighting length scale, metres
# torch 2.0.1 has no bfloat16 kernel for upsample_bilinear2d, which the TransFuser backbone uses in
# its fusion blocks, so autocast is off by default. fp32 measured at ~156 samples/s on the 4090.
AMP = os.environ.get("AMP", "0") == "1"


class _NoAutocast:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def amp_ctx():
    return torch.autocast("cuda", dtype=torch.bfloat16) if AMP else _NoAutocast()


IMG_CACHE = EXP / "bevsdf_images.npy"
ROW_BYTES = 256 * 1024 * 3


class BevSdfDataset(Dataset):
    """Camera feature + status + cached SDF target.

    Reads the pre-decoded (256, 1024, 3) uint8 stitch from bevsdf_images.npy when it exists, which is
    what makes training GPU-bound; otherwise falls back to decoding the three JPEGs per sample
    (~37 ms of CPU, caps the run at ~124 samples/s).
    """

    def __init__(self, idx, keep):
        self.rows = np.asarray(keep)
        self.token = idx["token"][keep]
        self.p = [idx["path_l0"][keep], idx["path_f0"][keep], idx["path_r0"][keep]]
        self.status = idx["status"][keep]
        self.use_cache = IMG_CACHE.exists()
        self.fh = None                          # opened lazily, per worker
        self.off = int(np.load(IMG_CACHE, mmap_mode="r").offset) if self.use_cache else 0

    def __len__(self):
        return len(self.token)

    def _stitch(self, i):
        ims = []
        for k, crop in enumerate([True, False, True]):
            im = cv2.imread(str(SENSOR_ROOT / self.p[k][i]))
            if im is None:
                raise FileNotFoundError(self.p[k][i])
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            ims.append(im[28:-28, 416:-416] if crop else im[28:-28])
        # identical to TransfuserFeatureBuilder._get_camera_feature
        return cv2.resize(np.concatenate(ims, axis=1), (1024, 256))

    def __getitem__(self, i):
        if self.use_cache:
            # pread, NOT mmap: random access across a 76 GiB file triggers kernel readahead that
            # saturates the SSD and collapses to 40 rows/s. pread + FADV_RANDOM measures 1368 rows/s.
            if self.fh is None:
                self.fh = os.open(IMG_CACHE, os.O_RDONLY)
                os.posix_fadvise(self.fh, 0, 0, os.POSIX_FADV_RANDOM)
            buf = os.pread(self.fh, ROW_BYTES, self.off + int(self.rows[i]) * ROW_BYTES)
            img = np.frombuffer(buf, np.uint8).reshape(256, 1024, 3)
        else:
            img = self._stitch(i)
        cam = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))    # uint8 CHW
        sdf = np.load(TARGETS / f"sdf_{self.token[i]}.npz")["sdf"].astype(np.float32)
        return cam, torch.from_numpy(self.status[i]), torch.from_numpy(sdf)


def build_model():
    cfg = TransfuserConfig()
    cfg.latent = True
    cfg.num_bev_classes = 1                                   # regression head
    model = TransfuserModel(TrajectorySampling(time_horizon=4, interval_length=0.5), cfg)
    sd = torch.load(CKPT_INIT, map_location="cpu")["state_dict"]
    sd = {k.replace("agent._transfuser_model.", ""): v for k, v in sd.items()
          if k.startswith("agent._transfuser_model.")}
    # the final 1x1 conv changes shape 7 -> 1; drop it and let it train from scratch
    drop = [k for k in sd if k.startswith("_bev_semantic_head.2.")]
    for k in drop:
        sd.pop(k)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"warm start from ltf_seed_0: dropped {drop}, missing {len(missing)}, unexpected {len(unexpected)}",
          flush=True)
    return model


def loss_fn(pred, tgt):
    """Smooth-L1 weighted towards the drivable boundary (that is the only part the energy reads)."""
    w = 1.0 + 4.0 * torch.exp(-tgt.abs() / BW)
    return (w * F.smooth_l1_loss(pred, tgt, reduction="none", beta=0.5)).mean()


@torch.no_grad()
def validate(model, loader, dev):
    model.eval()
    tot = dict(loss=0.0, mae=0.0, mae_near=0.0, sign=0.0, iou_n=0.0, iou_d=0.0, n=0, n_near=0)
    for cam, st, sdf in loader:
        cam = cam.to(dev, non_blocking=True).float().div_(255.0)   # uint8 -> float on GPU
        st, sdf = st.to(dev), sdf.to(dev, non_blocking=True)
        with amp_ctx():
            p = model({"camera_feature": cam, "status_feature": st})["bev_semantic_map"][:, 0]
        p = p.float()
        b = cam.shape[0]
        tot["loss"] += loss_fn(p, sdf).item() * b
        tot["mae"] += (p - sdf).abs().mean().item() * b
        near = sdf.abs() <= 1.0
        if near.any():
            tot["mae_near"] += (p - sdf).abs()[near].mean().item() * b
            tot["n_near"] += b
        tot["sign"] += ((p < 0) == (sdf < 0)).float().mean().item() * b
        inter = ((p < 0) & (sdf < 0)).sum().item(); union = ((p < 0) | (sdf < 0)).sum().item()
        tot["iou_n"] += inter; tot["iou_d"] += union
        tot["n"] += b
    n = max(tot["n"], 1)
    return dict(loss=tot["loss"] / n, mae=tot["mae"] / n,
                mae_near=tot["mae_near"] / max(tot["n_near"], 1),
                sign_acc=tot["sign"] / n, iou=tot["iou_n"] / max(tot["iou_d"], 1))


def main():
    idx = dict(np.load(EXP / "bevsdf_index_navtrain.npz", allow_pickle=False))
    have = np.array([(TARGETS / f"sdf_{t}.npz").exists() for t in idx["token"]])
    print(f"index {len(idx['token'])} samples, targets present for {have.sum()}", flush=True)

    # split by LOG: adjacent frames of one log are near-duplicates and would leak
    logs = np.unique(idx["log"][have])
    rng = np.random.default_rng(0)
    val_logs = set(rng.permutation(logs)[:max(1, int(len(logs) * VAL_FRAC))].tolist())
    is_val = np.array([l in val_logs for l in idx["log"]])
    tr_keep = np.where(have & ~is_val)[0]
    va_keep = np.where(have & is_val)[0]
    print(f"train {len(tr_keep)} samples / {len(logs)-len(val_logs)} logs | "
          f"val {len(va_keep)} samples / {len(val_logs)} logs", flush=True)

    # OpenCV spawns one thread per core inside every DataLoader worker; with 10 workers that is
    # ~340 threads on 32 cores and the GPU starves (measured: 101 -> 217 samples/s with this off).
    def winit(_):
        cv2.setNumThreads(0)
        torch.set_num_threads(1)

    dl = dict(num_workers=WORKERS, pin_memory=True, persistent_workers=WORKERS > 0,
              worker_init_fn=winit)
    tr = DataLoader(BevSdfDataset(idx, tr_keep), batch_size=BATCH, shuffle=True, drop_last=True,
                    prefetch_factor=4 if WORKERS else None, **dl)
    va = DataLoader(BevSdfDataset(idx, va_keep), batch_size=BATCH, shuffle=False, **dl)

    dev = "cuda"
    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * len(tr),
                                                pct_start=0.05)
    hist_path = OUT / "history.json"
    hist = json.load(open(hist_path)) if hist_path.exists() else []
    best = min([h["val_mae_near"] for h in hist], default=float("inf"))
    # resume: history alone would skip epochs while silently restarting from the warm-start weights,
    # and OneCycleLR would be at the wrong point, so restore the model and fast-forward the schedule
    if hist and (OUT / "last.pt").exists():
        ck = torch.load(OUT / "last.pt", map_location="cpu")
        model.load_state_dict(ck["model"])
        model.to(dev)
        for _ in range(len(hist) * len(tr)):
            sched.step()
        print(f"resumed from epoch {ck['epoch']} (best val MAE<1m so far {best:.3f} m)", flush=True)
    elif hist:
        raise SystemExit(f"{hist_path} has {len(hist)} epochs but no last.pt — refusing to resume blind")

    for ep in range(len(hist), EPOCHS):
        model.train()
        t0 = time.perf_counter(); run = 0.0; seen = 0
        for it, (cam, st, sdf) in enumerate(tr):
            cam = cam.to(dev, non_blocking=True).float().div_(255.0)
            st, sdf = st.to(dev), sdf.to(dev, non_blocking=True)
            with amp_ctx():
                p = model({"camera_feature": cam, "status_feature": st})["bev_semantic_map"][:, 0]
            loss = loss_fn(p.float(), sdf)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run += loss.item() * cam.shape[0]; seen += cam.shape[0]
            if (it + 1) % 100 == 0:
                print(f"ep {ep+1}/{EPOCHS} it {it+1}/{len(tr)} train_loss {run/seen:.4f} "
                      f"{seen/(time.perf_counter()-t0):.0f} samples/s", flush=True)
        train_loss = run / max(seen, 1)
        v = validate(model, va, dev)
        rec = dict(epoch=ep + 1, train_loss=train_loss, val_loss=v["loss"], val_mae=v["mae"],
                   val_mae_near=v["mae_near"], val_sign_acc=v["sign_acc"], val_iou=v["iou"],
                   lr=sched.get_last_lr()[0], mins=(time.perf_counter() - t0) / 60)
        hist.append(rec); json.dump(hist, open(hist_path, "w"), indent=1)
        print(f"EPOCH {ep+1}/{EPOCHS} train {train_loss:.4f} | val {v['loss']:.4f} "
              f"MAE {v['mae']:.3f} m | MAE<1m {v['mae_near']:.3f} m | sign {v['sign_acc']:.4f} "
              f"| IoU {v['iou']:.4f} | {rec['mins']:.1f} min", flush=True)
        torch.save({"model": model.state_dict(), "epoch": ep + 1, "rec": rec}, OUT / "last.pt")
        if v["mae_near"] < best:
            best = v["mae_near"]
            torch.save({"model": model.state_dict(), "epoch": ep + 1, "rec": rec}, OUT / "best.pt")
            print(f"  new best (val MAE within 1 m of the edge = {best:.3f} m)", flush=True)
    print(f"DONE best val MAE<1m {best:.3f} m -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
