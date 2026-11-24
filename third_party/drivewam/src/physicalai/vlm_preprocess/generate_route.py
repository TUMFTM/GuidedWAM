"""
Step 1 of the leak-free per-chunk navigation pipeline: classify the *route*
of each future chunk from GT ego pose alone.

A route is one of {"straight", "left", "right"}. It is derived from the net
yaw change of the ego between the chunk's first and last pose samples using
R_rel = R_start.T @ R_end,  yaw = atan2(R_rel[1,0], R_rel[0,0]).
    |yaw| < straight_deg (default 15°)  → "straight"
    yaw >=  straight_deg                → "left"
    yaw <= -straight_deg                → "right"

Route labels are a low-bandwidth signal comparable to a high-level nav
command ("at the next intersection, turn left") — analogous to what a real
routing system provides. They are safe to feed as conditioning to a
downstream world-action model even though they come from GT.

"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from physicalai.vlm_preprocess.generate_bev import (
    _chunk_key,
    _ego_pkl,
    _load_ego,
)


# ----------------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------------

def classify_route_from_yaw(yaw_deg: float, straight_deg: float = 15.0) -> str:
    if yaw_deg >= straight_deg:
        return "left"
    if yaw_deg <= -straight_deg:
        return "right"
    return "straight"


def compute_chunk_yaw_deg(
    ego: dict,
    chunk_idx: int,
    chunk_seconds: float = 4.0,
) -> Optional[float]:
    pose  = ego["ego_pose__world_from_rig"]   # (N, 4, 4)
    ts_us = ego["timestamps_us"]              # (N,)

    ts_s_all = (ts_us - ts_us[0]) / 1e6
    t0 = chunk_idx * chunk_seconds
    t1 = (chunk_idx + 1) * chunk_seconds

    mask = (ts_s_all >= t0) & (ts_s_all < t1 + 1e-6)
    if not mask.any():
        return None

    idxs = np.where(mask)[0]
    c0, c1 = int(idxs[0]), int(idxs[-1])

    R_start = pose[c0, :3, :3]
    R_end   = pose[c1, :3, :3]
    R_rel   = R_start.T @ R_end
    yaw_rad = float(np.arctan2(R_rel[1, 0], R_rel[0, 0]))
    return float(np.degrees(yaw_rad))


# ----------------------------------------------------------------------------
# Per-clip worker
# ----------------------------------------------------------------------------

def _process_one(
    args_tuple,
) -> Tuple[str, List[Tuple[int, str]]]:
    clip_id, dataset_root, num_chunks, chunk_seconds, straight_deg = args_tuple
    try:
        pkl = _ego_pkl(dataset_root, clip_id)
        if not os.path.exists(pkl):
            print(f"[WARN] {clip_id}: missing ego pkl")
            return (clip_id, [])

        ego = _load_ego(pkl)
        out: List[Tuple[int, str]] = []
        for ci in range(num_chunks):
            yaw_deg = compute_chunk_yaw_deg(ego, ci, chunk_seconds=chunk_seconds)
            if yaw_deg is None:
                print(f"[WARN] {clip_id} chunk {ci}: no pose samples in window")
                continue
            label = classify_route_from_yaw(yaw_deg, straight_deg)
            out.append((ci, label))
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
    parser.add_argument("--straight_deg", type=float, default=15.0,
        help="Absolute yaw change (deg) below which a chunk is 'straight'.")
    parser.add_argument("--clip_ids_file", default=None,
        help="Path to a JSON file containing a list of clip_ids.")
    parser.add_argument("--route_store_name", default="route_store.json",
        help="Output filename (under output_dir) for {chunk_key: label}.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_store  = out_dir / args.route_store_name

    if args.clip_ids_file:
        with open(args.clip_ids_file) as f:
            clip_ids = list(json.load(f))
    else:
        from physicalai.vlm_preprocess.generate_scene_evolving_guidance import select_demo_clips
        clip_ids = select_demo_clips(
            args.clip_index, args.dataset_root, args.num_clips, seed=args.seed
        )
    print(f"[main] {len(clip_ids)} clips, {args.num_workers} workers → {out_store}")

    worker_args = [
        (cid, args.dataset_root, args.num_chunks, args.chunk_seconds, args.straight_deg)
        for cid in clip_ids
    ]

    ok_clip_ids: List[str] = []
    ok_chunk_keys: List[str] = []
    label_map: Dict[str, str] = {}

    if args.num_workers == 1:
        results_iter = (_process_one(a) for a in worker_args)
        pool = None
    else:
        pool = Pool(processes=args.num_workers)
        results_iter = pool.imap_unordered(_process_one, worker_args)

    try:
        for clip_id, chunks in tqdm(
            results_iter, total=len(clip_ids), desc="routes", unit="clip"
        ):
            if not chunks:
                continue
            ok_clip_ids.append(clip_id)
            for ci, label in chunks:
                key = _chunk_key(clip_id, ci)
                label_map[key] = label
                ok_chunk_keys.append(key)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    with open(out_store, "w") as f:
        json.dump(label_map, f, indent=2)

    # Distribution summary
    from collections import Counter
    counts = Counter(label_map.values())
    total = sum(counts.values())
    print(f"[main] {len(ok_clip_ids)}/{len(clip_ids)} clips, {len(ok_chunk_keys)} chunks")
    print(f"[main] labels → {out_store}")
    for lab in ("straight", "left", "right"):
        n = counts.get(lab, 0)
        pct = 100 * n / max(1, total)
        print(f"        {lab:<8s}: {n:6d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
