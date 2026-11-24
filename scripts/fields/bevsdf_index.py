"""Build a compact training index for the BEV signed-distance net.

The navtrain SceneLoader costs ~65 s and ~10 GB of RAM, which cannot be paid per DataLoader worker.
This dumps everything a worker actually needs: the three front camera JPEG paths of the current
frame and the 8-dim status vector, plus the log name so the train/val split can be made by LOG
(adjacent frames of one log are near-duplicates and would leak across the split).

Writes bevsdf_index_<split>.npz: token, log, path_l0/f0/r0 (relative to the sensor root), status (N, 8).
"""
import os
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

import navsim
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader
from guidedwam import paths

ROOT = paths.NAVSIM_DATA
NAVSIM_V1 = paths.NAVSIM
SPLIT = os.environ.get("SPLIT", "navtrain")
OUT = Path(str(paths.EXP / f"bevsdf_index_{SPLIT}.npz"))
FRAME_IDX = 3                                    # TransFuser uses the current frame (history index 3)

cfg = OmegaConf.load(NAVSIM_V1 / f"navsim/planning/script/config/common/train_test_split/scene_filter/{SPLIT}.yaml")
loader = SceneLoader(data_path=ROOT / "navsim_logs/trainval",
                     original_sensor_path=ROOT / "sensor_blobs/trainval",
                     scene_filter=instantiate(cfg), synthetic_sensor_path=None,
                     synthetic_scenes_path=None, sensor_config=SensorConfig.build_no_sensors())

tok, log, p0, p1, p2, st = [], [], [], [], [], []
n_skip = 0
for t in sorted(loader.tokens):
    fr = loader.scene_frames_dicts[t][FRAME_IDX]
    cams = fr["cams"]
    try:
        a, b, c = (cams["CAM_L0"]["data_path"], cams["CAM_F0"]["data_path"], cams["CAM_R0"]["data_path"])
    except KeyError:
        n_skip += 1
        continue
    dyn = np.asarray(fr["ego_dynamic_state"], dtype=np.float32)
    tok.append(t); log.append(fr["log_name"]); p0.append(a); p1.append(b); p2.append(c)
    st.append(np.concatenate([np.asarray(fr["driving_command"], np.float32), dyn[:2], dyn[2:4]]))

np.savez_compressed(OUT, token=np.array(tok), log=np.array(log), path_l0=np.array(p0),
                    path_f0=np.array(p1), path_r0=np.array(p2), status=np.stack(st).astype(np.float32))
print(f"{SPLIT}: {len(tok)} samples, {len(set(log))} logs, skipped {n_skip} -> {OUT}", flush=True)
