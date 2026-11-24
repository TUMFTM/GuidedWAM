import torch
from easydict import EasyDict

drivewam_physicalai_cfg = EasyDict(__name__="Config: DriveWAM on PhysicalAI Autonomous Vehicles")

drivewam_physicalai_cfg.param_dtype = torch.bfloat16
drivewam_physicalai_cfg.save_root = './train_physicalAI'
drivewam_physicalai_cfg.patch_size = (1, 2, 2)

drivewam_physicalai_cfg.wan22_pretrained_model_name_or_path = "lingbot-va/lingbot-va-base"

# ----- Dataset -----
drivewam_physicalai_cfg.dataset_path = "./data/physicalai/front"
drivewam_physicalai_cfg.clip_index_path = "./data/physicalai/clip_index.parquet"

# ----- Training -----
drivewam_physicalai_cfg.split = "train"
drivewam_physicalai_cfg.max_clips = 100000
drivewam_physicalai_cfg.target_video_fps = 1
drivewam_physicalai_cfg.sample_duration_s = 12

drivewam_physicalai_cfg.enable_wandb = True
drivewam_physicalai_cfg.load_worker = 8
drivewam_physicalai_cfg.save_interval = 5000
drivewam_physicalai_cfg.gc_interval = 50

drivewam_physicalai_cfg.learning_rate = 1e-5
drivewam_physicalai_cfg.beta1 = 0.9
drivewam_physicalai_cfg.beta2 = 0.95
drivewam_physicalai_cfg.weight_decay = 0.1
drivewam_physicalai_cfg.warmup_steps = 10
drivewam_physicalai_cfg.batch_size = 1
drivewam_physicalai_cfg.gradient_accumulation_steps = 1
drivewam_physicalai_cfg.num_steps = 50000

# ----- Video / latent -----
drivewam_physicalai_cfg.height = 256
drivewam_physicalai_cfg.width = 448
drivewam_physicalai_cfg.text_max_length = 512
drivewam_physicalai_cfg.frame_chunk_size = 1
drivewam_physicalai_cfg.attn_window = 24
drivewam_physicalai_cfg.snr_shift = 5.0
drivewam_physicalai_cfg.action_snr_shift = 1.0

# ----- Action -----
drivewam_physicalai_cfg.action_dim = 30
drivewam_physicalai_cfg.future_action_steps = 120
drivewam_physicalai_cfg.action_chunk_count = 4
drivewam_physicalai_cfg.action_chunk_steps = 40

drivewam_physicalai_cfg.prompt = (
    "An autonomous vehicle navigating real-world roads safely, "
    "maintaining lane discipline and avoiding all collisions with surrounding vehicles and obstacles."
)

drivewam_physicalai_cfg.used_action_channel_ids = [0, 1, 2]  # dx, dy, dheading

drivewam_physicalai_cfg.action_norm_method = "quantiles"
drivewam_physicalai_cfg.norm_stat = {
    "q01": [-0.50788880, -1.31925802, -0.03548078],
    "q99": [3.50935863, 1.24382361, 0.03866158],
}

drivewam_physicalai_cfg.enable_text_encoder_cache = False
drivewam_physicalai_cfg.prompts_dir = "./data/physicalai/nav_prompts"