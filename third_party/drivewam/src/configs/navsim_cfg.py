import torch
from easydict import EasyDict

drivewam_navsim_cfg = EasyDict(__name__="Config: DriveWAM on navsim")

drivewam_navsim_cfg.param_dtype = torch.bfloat16
drivewam_navsim_cfg.save_root = './train_navsim'
drivewam_navsim_cfg.patch_size = (1, 2, 2)

drivewam_navsim_cfg.wan22_pretrained_model_name_or_path = (
    "./weights/lingbot-va-base"
)

drivewam_navsim_cfg.enable_wandb = True
drivewam_navsim_cfg.load_worker = 8
drivewam_navsim_cfg.save_interval = 5000
drivewam_navsim_cfg.gc_interval = 50

drivewam_navsim_cfg.learning_rate = 1e-5
drivewam_navsim_cfg.beta1 = 0.9
drivewam_navsim_cfg.beta2 = 0.95
drivewam_navsim_cfg.weight_decay = 0.1
drivewam_navsim_cfg.warmup_steps = 10
drivewam_navsim_cfg.batch_size = 1
drivewam_navsim_cfg.gradient_accumulation_steps = 1
drivewam_navsim_cfg.num_steps = 100000
drivewam_navsim_cfg.lr_decay_milestones = [50000, 70000, 90000]
drivewam_navsim_cfg.lr_decay_gamma = 0.5

drivewam_navsim_cfg.height = 256
drivewam_navsim_cfg.width = 448
drivewam_navsim_cfg.text_max_length = 512

drivewam_navsim_cfg.action_dim = 30
drivewam_navsim_cfg.future_action_steps = 8
drivewam_navsim_cfg.action_chunk_count = 2
drivewam_navsim_cfg.action_chunk_steps = 8
drivewam_navsim_cfg.action_per_frame = drivewam_navsim_cfg.action_chunk_steps
drivewam_navsim_cfg.future_image_indices = [1, 3, 5, 7]
drivewam_navsim_cfg.history_num_frames = 4
drivewam_navsim_cfg.frame_chunk_size = 1
drivewam_navsim_cfg.attn_window = 30

drivewam_navsim_cfg.snr_shift = 5.0
drivewam_navsim_cfg.action_snr_shift = 1.0

drivewam_navsim_cfg.command_prompts = [
    "autonomous driving scene, the intended route is to the left.",
    "autonomous driving scene, the intended route is straight.",
    "autonomous driving scene, the intended route is to the right.",
    "autonomous driving scene, the driving command is unknown.",
]

drivewam_navsim_cfg.used_action_channel_ids = [0, 1, 2]  # dx, dy, dheading
inverse_used_action_channel_ids = [len(drivewam_navsim_cfg.used_action_channel_ids)] * drivewam_navsim_cfg.action_dim
for i, j in enumerate(drivewam_navsim_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
drivewam_navsim_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

drivewam_navsim_cfg.action_norm_method = "quantiles"
drivewam_navsim_cfg.norm_stat = {
    "q01": [-0.01582209, -1.76743735, -0.18790549],
    "q99": [6.15755262, 2.46998216, 0.18174374],
}

drivewam_navsim_cfg.enable_text_encoder_cache = True
