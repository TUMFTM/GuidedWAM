"""Extract 10Hz mp4 videos and ego info (camera pose, velocity, etc.) for all clips.

Reads the raw PhysicalAI-AV dataset through the physical_ai_av devkit
(https://github.com/NVlabs/physical_ai_av) from a completed local download --
no Hugging Face hub access. Writes per clip:

    <output-dir>/<clip_id>/camera_front_wide_120fov.mp4      (10Hz video)
    <output-dir>/<clip_id>/camera_front_wide_120fov_ego.pkl  (ego + calibration)

Processing finishes with a cam_trajectory_samples.png visualization of the
first few clips.

Usage:
    python -m src.physicalai.process_data --num_workers 16
    python -m src.physicalai.process_data --output_dir /path/to/output
"""

import argparse
import contextlib
import multiprocessing as mp
import os
import pathlib
import pickle
import traceback

import cv2
import numpy as np
import physical_ai_av
from physical_ai_av.utils import hf_interface

os.environ["HF_HUB_OFFLINE"] = "1"

CAM_NAME = "camera_front_wide_120fov"
CAMERA_PERIOD_US = 100_000  # 10Hz
OUTPUT_FPS = 10


def patch_local_only_hf_interface() -> None:
    """Force HfRepoInterface to read from local_dir only, never the hub."""

    def offline_download_file(self, filename, **kwargs):
        local_path = pathlib.Path(self.local_dir) / filename if self.local_dir else None
        if local_path and local_path.exists():
            return str(local_path)
        raise FileNotFoundError(f"[offline strict] missing file in local_dir: {filename}")

    def offline_is_file_cached(self, filename):
        local_path = pathlib.Path(self.local_dir) / filename if self.local_dir else None
        return bool(local_path and local_path.exists())

    @contextlib.contextmanager
    def offline_open_file(self, filename, mode="rb", maybe_stream=False):
        local_path = pathlib.Path(self.local_dir) / filename if self.local_dir else None
        if local_path and local_path.exists():
            with open(local_path, mode) as f:
                yield f
            return
        raise FileNotFoundError(f"[offline strict] missing file in local_dir: {filename}")

    hf_interface.HfRepoInterface.download_file = offline_download_file
    hf_interface.HfRepoInterface.is_file_cached = offline_is_file_cached
    hf_interface.HfRepoInterface.open_file = offline_open_file


def nearest_indices(sorted_ref_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    idx_right = np.searchsorted(sorted_ref_ts, query_ts, side="left")
    idx_right = np.clip(idx_right, 0, len(sorted_ref_ts) - 1)
    idx_left = np.clip(idx_right - 1, 0, len(sorted_ref_ts) - 1)
    pick_right = np.abs(sorted_ref_ts[idx_right] - query_ts) < np.abs(
        sorted_ref_ts[idx_left] - query_ts
    )
    return np.where(pick_right, idx_right, idx_left)


def select_camera_10hz_indices(cam_ts: np.ndarray, period_us: int = CAMERA_PERIOD_US) -> np.ndarray:
    if len(cam_ts) == 0:
        return np.array([], dtype=np.int64)
    targets = np.arange(cam_ts[0], cam_ts[-1] + period_us, period_us, dtype=np.int64)
    idx = nearest_indices(cam_ts.astype(np.int64), targets)
    return np.unique(idx.astype(np.int64))


def extract_camera_intrinsics(cam_model) -> dict:
    """Extract camera intrinsics into a pickle-friendly dict."""
    result = {
        "width": cam_model.width,
        "height": cam_model.height,
        "principal_point": np.array(cam_model.principal_point, dtype=np.float64),
    }
    # Store polynomial coefficients if available
    try:
        result["r2th"] = str(cam_model.r2th)
    except Exception:
        pass
    try:
        result["th2r"] = str(cam_model.th2r)
    except Exception:
        pass
    return result


def process_clip(avdi, clip_id: str, output_dir: str) -> bool:
    """Process a single clip: extract 10Hz mp4 + ego pkl.

    Returns True if successful, False if skipped/failed.
    """
    clip_dir = os.path.join(output_dir, clip_id)
    mp4_path = os.path.join(clip_dir, f"{CAM_NAME}.mp4")
    pkl_path = os.path.join(clip_dir, f"{CAM_NAME}_ego.pkl")

    # Skip if already done
    if os.path.exists(mp4_path) and os.path.exists(pkl_path):
        return True

    # Load camera video
    cam_feature = avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV
    video = avdi.get_clip_feature(clip_id, cam_feature, maybe_stream=False)

    cam_ts_all = video.timestamps.astype(np.int64)
    if len(cam_ts_all) == 0:
        video.close()
        return False

    # Select 10Hz frame indices
    keep_idx = select_camera_10hz_indices(cam_ts_all, period_us=CAMERA_PERIOD_US)
    keep_ts = cam_ts_all[keep_idx]

    if len(keep_idx) == 0:
        video.close()
        return False

    # Decode 10Hz frames
    frames = video.decode_images_from_frame_indices(keep_idx.astype(np.int64))
    h, w = frames.shape[1], frames.shape[2]
    video.close()

    # Load egomotion
    ego_interp = avdi.get_clip_feature(
        clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=False
    )

    # Load calibration
    extr = avdi.get_clip_feature(
        clip_id, avdi.features.CALIBRATION.SENSOR_EXTRINSICS, maybe_stream=False
    )
    intr = avdi.get_clip_feature(
        clip_id, avdi.features.CALIBRATION.CAMERA_INTRINSICS, maybe_stream=False
    )

    rig_T_cam = extr.sensor_poses[CAM_NAME]  # static sensor extrinsics
    cam_model = intr.camera_models[CAM_NAME]

    # Query egomotion at each 10Hz timestamp
    ego_states = ego_interp(keep_ts)
    # ego_states.pose is a batched RigidTransform (world_T_rig)
    world_T_rig = ego_states.pose.as_matrix()  # (N, 4, 4)
    velocities = ego_states.velocity  # (N, 3)
    accelerations = ego_states.acceleration  # (N, 3)
    curvatures = ego_states.curvature  # (N, 1) or (N,)

    # Compute world_T_cam = world_T_rig @ rig_T_cam
    rig_T_cam_mat = rig_T_cam.as_matrix()  # (4, 4)
    world_T_cam = world_T_rig @ rig_T_cam_mat  # (N, 4, 4)

    # Build ego info dict
    ego_info = {
        "clip_id": clip_id,
        "camera_name": CAM_NAME,
        "timestamps_us": keep_ts,  # (N,) int64, timestamps of the 10Hz frames (us)
        "frame_indices": keep_idx,  # (N,) int64, frame indices into the original 30Hz video
        # ---- ego pose ----
        # Ego pose in the world frame, 4x4 homogeneous matrices.
        # Usage: pts_world = ego_pose @ pts_rig
        # rig frame = vehicle frame (x forward, y left, z up)
        "ego_pose__world_from_rig": world_T_rig.astype(np.float64),  # (N, 4, 4)
        # Camera pose in the world frame = ego_pose @ extrinsics_rig_from_cam
        # Usage: pts_world = cam_pose @ pts_cam
        "cam_pose__world_from_cam": world_T_cam.astype(np.float64),  # (N, 4, 4)
        # ---- ego dynamics (all in the rig/vehicle frame, x forward y left z up) ----
        "ego_velocity_rig": velocities.astype(np.float64),  # (N, 3), m/s
        "ego_acceleration_rig": accelerations.astype(np.float64),  # (N, 3), m/s^2
        # Path curvature = 1 / turn radius, in 1/m; positive = turning left
        "ego_curvature": np.asarray(curvatures, dtype=np.float64).reshape(-1),  # (N,)
        # ---- static calibration (constant over the clip) ----
        # Camera-to-rig extrinsics, 4x4 homogeneous matrix.
        # Usage: pts_rig = extrinsics @ pts_cam
        "extrinsics__rig_from_cam": rig_T_cam_mat.astype(np.float64),  # (4, 4)
        "camera_intrinsics": extract_camera_intrinsics(cam_model),
    }

    # Write output
    os.makedirs(clip_dir, exist_ok=True)

    # Write mp4 at 10fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mp4_path, fourcc, OUTPUT_FPS, (w, h))
    for i in range(len(frames)):
        bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()

    # Write pkl
    with open(pkl_path, "wb") as f:
        pickle.dump(ego_info, f, protocol=pickle.HIGHEST_PROTOCOL)

    return True


_worker_avdi = None  # per-process global, initialized by _worker_init


def _worker_init(data_root: str) -> None:
    global _worker_avdi
    patch_local_only_hf_interface()
    _worker_avdi = physical_ai_av.PhysicalAIAVDatasetInterface(
        local_dir=data_root, revision="main"
    )


def _worker_fn(args_tuple: tuple) -> str:
    """Worker function: returns 'done' / 'skip' / 'fail'."""
    clip_id, output_dir = args_tuple
    try:
        ok = process_clip(_worker_avdi, clip_id, output_dir)
        return "done" if ok else "skip"
    except FileNotFoundError:
        return "skip"
    except Exception:
        traceback.print_exc()
        return "fail"


def main_extract(args) -> None:
    import pandas as pd

    patch_local_only_hf_interface()

    clip_index = pd.read_parquet(os.path.join(args.dataset_root, "clip_index.parquet"))

    if args.clip_list:
        with open(args.clip_list) as f:
            clip_ids = [line.strip() for line in f if line.strip()]
    elif args.chunks is not None:
        if "-" in args.chunks:
            lo, hi = args.chunks.split("-", 1)
            chunk_range = range(int(lo), int(hi) + 1)
        else:
            chunk_range = range(int(args.chunks), int(args.chunks) + 1)
        clip_ids = clip_index[clip_index["chunk"].isin(chunk_range)].index.tolist()
    else:
        clip_ids = clip_index.index.tolist()

    total = len(clip_ids)
    print(f"Total clips to process: {total}, workers: {args.num_workers}")
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = [(cid, args.output_dir) for cid in clip_ids]
    done = 0
    skipped = 0
    failed = 0

    if args.num_workers <= 1:
        _worker_init(args.dataset_root)
        for i, t in enumerate(tasks):
            result = _worker_fn(t)
            if result == "done":
                done += 1
            elif result == "skip":
                skipped += 1
            else:
                failed += 1
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"[{i + 1}/{total}] done={done} skipped={skipped} failed={failed}")
    else:
        with mp.Pool(
            processes=args.num_workers,
            initializer=_worker_init,
            initargs=(args.dataset_root,),
        ) as pool:
            for i, result in enumerate(pool.imap_unordered(_worker_fn, tasks)):
                if result == "done":
                    done += 1
                elif result == "skip":
                    skipped += 1
                else:
                    failed += 1
                if (i + 1) % 10 == 0 or (i + 1) == total:
                    print(f"[{i + 1}/{total}] done={done} skipped={skipped} failed={failed}")

    # Plot the camera trajectories of the first few clips as a sanity check
    visualize_samples(args.output_dir, n_samples=min(4, done))


def visualize_samples(output_dir: str, n_samples: int = 4) -> None:
    """Plot cam_pose trajectories for the first few processed clips."""
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pkl_paths = sorted(glob.glob(os.path.join(output_dir, "*", f"{CAM_NAME}_ego.pkl")))
    if not pkl_paths:
        print("No pkl files found, skip visualization.")
        return

    n_samples = min(n_samples, len(pkl_paths))
    if n_samples <= 0:
        return
    pkl_paths = pkl_paths[:n_samples]

    fig, axes = plt.subplots(1, n_samples, figsize=(6 * n_samples, 6), squeeze=False)
    axes = axes[0]

    for ax, pkl_path in zip(axes, pkl_paths):
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)

        ego_poses = d["ego_pose__world_from_rig"]  # (N, 4, 4)

        # BEV view anchored at the first frame: map the world-frame trajectory
        # into the first frame's ego frame, so the ego starts at the origin
        # heading right.
        T0_inv = np.linalg.inv(ego_poses[0])  # world → ego_frame_0
        pos_world = ego_poses[:, :3, 3]  # (N, 3) world translation of every frame
        # Into the first frame's local frame: x forward, y left, z up
        pos_local = (T0_inv[:3, :3] @ pos_world.T).T + T0_inv[:3, 3]  # (N, 3)

        # BEV: forward (x_local) points right, left (y_local) points up
        bev_x = pos_local[:, 0]
        bev_y = pos_local[:, 1]

        ax.plot(bev_x, bev_y, "b-", linewidth=1.5)
        ax.plot(0, 0, "go", markersize=10, label="start")
        ax.plot(bev_x[-1], bev_y[-1], "r^", markersize=8, label="end")

        # Heading arrows: each frame's ego x-axis (forward) in the local frame
        step = max(1, len(ego_poses) // 10)
        for i in range(0, len(ego_poses), step):
            fwd_world = ego_poses[i, :3, 0]  # ego x-axis in world
            fwd_local = T0_inv[:3, :3] @ fwd_world
            scale = 0.5
            ax.arrow(
                bev_x[i], bev_y[i], fwd_local[0] * scale, fwd_local[1] * scale,
                head_width=0.15, head_length=0.08, fc="orange", ec="orange",
            )

        clip_id = d["clip_id"]
        ax.set_title(clip_id[:12] + "...", fontsize=10)
        ax.set_xlabel("forward (m)")
        ax.set_ylabel("left (m)")
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Camera trajectory (world frame, XY plane)", fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "cam_trajectory_samples.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved visualization to {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Extract 10Hz video + ego info for all clips")
    parser.add_argument(
        "--dataset_root",
        default="./data/physicalai",
        help="Local download of the raw Hugging Face dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="./data/physicalai/front",
        help="Where to write per-clip mp4 + ego pkl",
    )
    parser.add_argument("--chunks", default=None, help="e.g. '0-5' or '3'")
    parser.add_argument("--clip_list", default=None, help="text file, one clip_id per line")
    parser.add_argument("--num_workers", type=int, default=16, help="number of parallel workers")
    args = parser.parse_args()
    main_extract(args)


if __name__ == "__main__":
    main()
