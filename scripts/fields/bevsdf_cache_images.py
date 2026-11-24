"""Pre-decode the training images once into a single uint8 memmap.

Measured: decoding 3x 1920x1080 JPEGs and building the 5760x1024 stitch per sample costs ~37 ms of
CPU and stops scaling at ~124 samples/s across 32 cores (memory-bandwidth bound on the intermediate
buffers), below the GPU's 156 samples/s. Doing it once turns every epoch into a 786 KB sequential
read per sample and makes training GPU-bound.

Output: bevsdf_images.npy, shape (N, 256, 1024, 3) uint8, row i = index row i of
bevsdf_index_navtrain.npz, ~81 GB. Shards write disjoint row ranges, so they are safe in parallel.

Env: SHARD NSHARD
"""
import os
from pathlib import Path

import cv2
import numpy as np
from guidedwam import paths

cv2.setNumThreads(0)
EXP = paths.EXP
SENSOR_ROOT = paths.NAVSIM_DATA / "sensor_blobs/trainval"
CACHE = EXP / "bevsdf_images.npy"
SHARD = int(os.environ.get("SHARD", "0"))
NSHARD = int(os.environ.get("NSHARD", "1"))

idx = dict(np.load(EXP / "bevsdf_index_navtrain.npz", allow_pickle=False))
N = len(idx["token"])
if SHARD == 0 and not CACHE.exists():
    np.lib.format.open_memmap(CACHE, mode="w+", dtype=np.uint8, shape=(N, 256, 1024, 3))
    print(f"allocated {CACHE} for {N} samples", flush=True)
while not CACHE.exists():
    import time; time.sleep(2)

arr = np.load(CACHE, mmap_mode="r+")
done_flag = EXP / f"bevsdf_images_done_{SHARD}_{NSHARD}.txt"
rows = range(SHARD, N, NSHARD)
n = 0
for i in rows:
    try:
        ims = []
        for key, crop in (("path_l0", True), ("path_f0", False), ("path_r0", True)):
            im = cv2.imread(str(SENSOR_ROOT / idx[key][i]))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            ims.append(im[28:-28, 416:-416] if crop else im[28:-28])
        arr[i] = cv2.resize(np.concatenate(ims, axis=1), (1024, 256))
        n += 1
    except Exception as e:
        print(f"FAILED row {i} ({idx['token'][i]}): {type(e).__name__}: {e}", flush=True)
    if n % 2000 == 0:
        print(f"[shard {SHARD}] {n}/{len(rows)}", flush=True)
arr.flush()
done_flag.write_text(str(n))
print(f"[shard {SHARD}] DONE {n}/{len(rows)}", flush=True)
