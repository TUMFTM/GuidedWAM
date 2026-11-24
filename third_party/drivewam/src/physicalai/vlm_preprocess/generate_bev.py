"""
Generate per-chunk BEV trajectory visualizations for AV clips.

Each ~20 s clip is split into non-overlapping windows (default 5 × 4 s).
For each chunk we render a
BEV image whose origin (0, 0) is the ego pose at the *chunk's* start,
X = forward (longitudinal), Y = lateral (left positive).

All BEV images for all (clip_id, chunk_idx) pairs are written to a single
compressed .npz archive (bev_store.npz). Keys are flat strings built by
``_chunk_key(clip_id, chunk_idx)``.

"""
from __future__ import annotations

import argparse
import io
import os
import pickle
import sys
import traceback
import zipfile
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _chunk_key(clip_id: str, chunk_idx: int) -> str:
    """Flat key for storing per-chunk arrays in npz / dicts."""
    return f"{clip_id.replace('/', '__')}__c{chunk_idx}"


# ----------------------------------------------------------------------------
# Ego pkl loading
# ----------------------------------------------------------------------------

def _ego_pkl(dataset_root: str, clip_id: str) -> str:
    return os.path.join(dataset_root, clip_id, "camera_front_wide_120fov_ego.pkl")


def _load_ego(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------
# Trajectory processing (per-chunk, anchored at the chunk's start pose)
# ----------------------------------------------------------------------------

def process_trajectory_chunk(
    ego: dict,
    chunk_idx: int,
    chunk_seconds: float = 4.0,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], np.ndarray]]:
    """Return trajectory for one chunk, anchored at that chunk's start pose.

    Returns (xy_local, ts_s, speeds, curvature, marker_idxs, marker_times), or
    None if the chunk window contains no pose samples.

    xy_local uses the rig frame at the *chunk start* as reference:
      X = forward, Y = left (lateral positive).
    ts_s is relative to the chunk start (ranges roughly [0, chunk_seconds]).
    """
    pose  = ego["ego_pose__world_from_rig"]   # (N, 4, 4)
    ts_us = ego["timestamps_us"]              # (N,)
    vel   = ego.get("ego_velocity_rig")       # (N, 3) or None
    curv  = ego.get("ego_curvature")          # (N,)   or None

    ts_s_all = (ts_us - ts_us[0]) / 1e6
    t0 = chunk_idx * chunk_seconds
    t1 = (chunk_idx + 1) * chunk_seconds

    mask = (ts_s_all >= t0) & (ts_s_all < t1 + 1e-6)
    if not mask.any():
        return None

    idxs = np.where(mask)[0]
    c0 = int(idxs[0])

    # Anchor at the chunk-start pose (not the clip-start pose)
    xyz_world   = pose[:, :3, 3]                 # (N, 3)
    delta_world = xyz_world - xyz_world[c0]      # (N, 3) displacement from chunk start
    R0          = pose[c0, :3, :3]               # rig axes at chunk start, expressed in world (cols)
    delta_local = delta_world @ R0               # world → rig via R0 (orthogonal)
    xy_local    = delta_local[idxs, :2]          # (M, 2) [fwd, lateral]

    ts_s = ts_s_all[idxs] - ts_s_all[c0]         # relative to chunk start
    speeds = (
        np.linalg.norm(vel[idxs], axis=1) if vel is not None else np.zeros(len(idxs))
    )
    curvature = curv[idxs] if curv is not None else np.zeros(len(idxs))

    # 1-second markers within the chunk (chunk is only ~4 s)
    marker_times = np.arange(0.0, ts_s[-1] + 0.1, 1.0)
    marker_idxs = [int(np.argmin(np.abs(ts_s - t))) for t in marker_times]

    return xy_local, ts_s, speeds, curvature, marker_idxs, marker_times


# ----------------------------------------------------------------------------
# BEV rendering
# ----------------------------------------------------------------------------

def render_bev_to_array(
    xy_local: np.ndarray,
    speeds: np.ndarray,
    marker_idxs: List[int],
    chunk_idx: int,
    chunk_seconds: float = 4.0,
) -> np.ndarray:
    """Render BEV for one chunk; returns H×W×3 uint8 RGB array."""
    fig, ax = plt.subplots(figsize=(5, 5))

    max_spd = float(speeds.max()) if speeds.size and speeds.max() > 1e-3 else 1.0
    x, y = xy_local[:, 0], xy_local[:, 1]

    if len(x) >= 2:
        pts  = np.stack([x, y], axis=1)[:, np.newaxis, :]
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        seg_spd = (speeds[:-1] + speeds[1:]) / 2
        norm = plt.Normalize(0, max_spd)
        lc = mc.LineCollection(segs, cmap="viridis", norm=norm, linewidths=2.5, zorder=2)
        lc.set_array(seg_spd)
        ax.add_collection(lc)

        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.03)
        cbar.set_label("speed (m/s)", fontsize=8)

    # Chunk-start ego: green square
    ax.plot(0, 0, "gs", markersize=10, label="Start", zorder=4)

    # 1s markers: dots only
    for mi in marker_idxs:
        if 0 <= mi < len(x):
            px, py = xy_local[mi]
            ax.plot(px, py, "o", color="white", markersize=7,
                    markeredgecolor="black", markeredgewidth=0.8, zorder=5)

    if len(x) > 0:
        ax.plot(x[-1], y[-1], "rs", markersize=8, label="End", zorder=4)

    all_x = np.concatenate([x, [0.0]]) if len(x) else np.array([0.0])
    all_y = np.concatenate([y, [0.0]]) if len(y) else np.array([0.0])
    margin = 5
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("X / forward (m)")
    ax.set_ylabel("Y / lateral (m)")
    t0 = chunk_idx * chunk_seconds
    t1 = (chunk_idx + 1) * chunk_seconds
    ax.set_title(f"BEV Chunk {chunk_idx}  [{t0:.0f}-{t1:.0f}s]")

    plt.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()
    plt.close(fig)
    return img


# ----------------------------------------------------------------------------
# Per-clip worker (renders all chunks for a single clip)
# ----------------------------------------------------------------------------

def _process_one(args_tuple) -> Tuple[str, List[Tuple[int, np.ndarray]]]:
    """Return (clip_id, [(chunk_idx, bev_img_array), ...])."""
    clip_id, dataset_root, num_chunks, chunk_seconds = args_tuple
    try:
        pkl = _ego_pkl(dataset_root, clip_id)
        if not os.path.exists(pkl):
            print(f"[WARN] {clip_id}: missing ego pkl")
            return (clip_id, [])

        ego = _load_ego(pkl)
        out: List[Tuple[int, np.ndarray]] = []
        for ci in range(num_chunks):
            result = process_trajectory_chunk(ego, ci, chunk_seconds=chunk_seconds)
            if result is None:
                print(f"[WARN] {clip_id} chunk {ci}: no pose samples in window")
                continue
            xy_local, ts_s, speeds, _, marker_idxs, _ = result
            img = render_bev_to_array(xy_local, speeds, marker_idxs, ci, chunk_seconds)
            out.append((ci, img))
        print(f"[OK] {clip_id} -> {len(out)}/{num_chunks} chunks")
        return (clip_id, out)

    except Exception:
        print(f"[ERR] {clip_id}:\n{traceback.format_exc()}")
        return (clip_id, [])


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root",
        default="./data/physicalai/front")
    parser.add_argument("--clip_index",
        default="./data/physicalai/clip_index.parquet")
    parser.add_argument("--output_dir",
        default=None)
    parser.add_argument("--num_clips", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=5)
    parser.add_argument("--chunk_seconds", type=float, default=4.0)
    parser.add_argument("--clip_ids_file", default=None,
        help="Path to a JSON file containing a list of clip_ids.")
    parser.add_argument("--bev_store_name", default="bev_store.npz")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / args.bev_store_name

    if args.clip_ids_file:
        import json as _json
        with open(args.clip_ids_file) as f:
            clip_ids = list(_json.load(f))
    else:
        from physicalai.vlm_preprocess.generate_scene_evolving_guidance import select_demo_clips
        clip_ids = select_demo_clips(
            args.clip_index, args.dataset_root, args.num_clips, seed=args.seed
        )
    print(f"[main] {len(clip_ids)} clips, {args.num_workers} workers → {out_npz}")

    worker_args = [
        (cid, args.dataset_root, args.num_chunks, args.chunk_seconds)
        for cid in clip_ids
    ]

    ok_clip_ids: List[str] = []
    ok_chunk_keys: List[str] = []
    with zipfile.ZipFile(out_npz, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if args.num_workers == 1:
            results_iter = (_process_one(a) for a in worker_args)
            pool = None
        else:
            pool = Pool(processes=args.num_workers)
            results_iter = pool.imap_unordered(_process_one, worker_args)

        try:
            for clip_id, chunks in tqdm(
                results_iter, total=len(clip_ids), desc="BEV chunks", unit="clip"
            ):
                if not chunks:
                    continue
                ok_clip_ids.append(clip_id)
                for ci, img in chunks:
                    key = _chunk_key(clip_id, ci)
                    buf = io.BytesIO()
                    np.save(buf, img)
                    zf.writestr(f"{key}.npy", buf.getvalue())
                    ok_chunk_keys.append(key)
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    print(f"[main] {len(ok_clip_ids)}/{len(clip_ids)} clips, "
          f"{len(ok_chunk_keys)} chunks → {out_npz}")


if __name__ == "__main__":
    main()
