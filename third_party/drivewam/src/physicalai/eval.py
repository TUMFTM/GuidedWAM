import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import VA_CONFIGS
from dataset.physicalai_dataset import PhysicalAITestDataset
from modules.physicalai_model import WanTransformer3DPhysicalAIModel
from modules.utils import load_text_encoder, load_tokenizer, load_vae
from utils import FlowMatchScheduler, data_seq_to_patch, get_mesh_id, init_logger, logger


# ===========================================================================
# Evaluator
# ===========================================================================


class PhysicalAIEvaluator:
    def __init__(self, config, checkpoint_path, attn_mode="torch", device=None):
        if attn_mode == "flex":
            raise ValueError(
                "Rollout evaluation uses single-stream inference. "
                "Use attn_mode='torch' or 'flashattn', not 'flex'."
            )

        self.config = config
        self.device = torch.device(device) if device is not None else torch.device("cuda:0")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size
        self.cache_name = "physicalai_eval_rollout"

        # Action denormalization parameters
        self.q01 = torch.tensor(config.norm_stat["q01"][:2], dtype=torch.float32)
        self.q99 = torch.tensor(config.norm_stat["q99"][:2], dtype=torch.float32)
        self.action_chunk_steps = config.action_chunk_steps

        logger.info(f"Loading transformer from {checkpoint_path} with attn_mode={attn_mode} ...")
        transformer_path = os.path.join(checkpoint_path, "transformer")
        self.transformer = WanTransformer3DPhysicalAIModel.from_pretrained(
            transformer_path,
            torch_dtype=self.dtype,
            attn_mode=attn_mode,
        ).to(self.device).eval()
        self.transformer.requires_grad_(False)

        logger.info("Loading VAE ...")
        self.vae = load_vae(
            os.path.join(config.wan22_pretrained_model_name_or_path, "vae"),
            torch_dtype=self.dtype,
            torch_device=self.device,
        ).eval()
        self.vae.requires_grad_(False)

        logger.info("Loading text encoder ...")
        self.tokenizer = load_tokenizer(
            os.path.join(config.wan22_pretrained_model_name_or_path, "tokenizer")
        )
        self.text_encoder = load_text_encoder(
            os.path.join(config.wan22_pretrained_model_name_or_path, "text_encoder"),
            torch_dtype=self.dtype,
            torch_device=self.device,
        ).eval()
        self.text_encoder.requires_grad_(False)

        self.video_scheduler = FlowMatchScheduler(
            shift=config.snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.action_scheduler = FlowMatchScheduler(
            shift=config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )

        self._prompt_cache = {}

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_images(self, images):
        """images: (B, T, C, H, W) uint8 -> normalized latents (B, C, T, H', W')."""
        batch_size, num_frames = images.shape[:2]
        imgs = F.interpolate(
            images.float().flatten(0, 1),
            size=(self.config.height, self.config.width),
            mode="bilinear",
            align_corners=False,
        )
        imgs = rearrange(imgs, "(b t) c h w -> b t c h w", b=batch_size, t=num_frames)
        videos = rearrange(imgs / 255.0 * 2.0 - 1.0, "b t c h w -> b c t h w").to(self.dtype)

        mu = self.vae.encode(videos).latent_dist.mean
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=mu.device, dtype=mu.dtype).view(
            1, -1, 1, 1, 1
        )
        latents_std = torch.tensor(self.vae.config.latents_std, device=mu.device, dtype=mu.dtype).view(
            1, -1, 1, 1, 1
        )
        return ((mu.float() - latents_mean) / latents_std).to(self.dtype)

    @torch.no_grad()
    def _encode_single_prompt(self, prompt: str) -> torch.Tensor:
        """Encode a single string, cache by exact string, return (seq_len, dim)."""
        if prompt not in self._prompt_cache:
            cleaned_prompt = prompt_clean(prompt) if prompt else ""
            inputs = self.tokenizer(
                [cleaned_prompt],
                padding="max_length",
                max_length=self.config.text_max_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            text_encoder_device = next(self.text_encoder.parameters()).device
            prompt_embeds = self.text_encoder(
                input_ids.to(text_encoder_device),
                attention_mask.to(text_encoder_device),
            ).last_hidden_state
            prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)
            seq_len = attention_mask.gt(0).sum().item()
            valid_tokens = prompt_embeds[0, :seq_len]
            pad_tokens = valid_tokens.new_zeros(
                self.config.text_max_length - seq_len,
                valid_tokens.shape[-1],
            )
            self._prompt_cache[prompt] = torch.cat([valid_tokens, pad_tokens], dim=0)
        return self._prompt_cache[prompt]

    @torch.no_grad()
    def _encode_prompt(self, prompts):
        """Encode a batch of positive prompts."""
        return torch.stack([self._encode_single_prompt(p) for p in prompts], dim=0)

    # ------------------------------------------------------------------
    # Input construction helpers
    # ------------------------------------------------------------------

    def _rng_devices(self):
        if self.device.type == "cuda" and self.device.index is not None:
            return [self.device.index]
        return []

    def _make_video_input(self, latents, timesteps, prompt_emb, ego_cond, frame_index):
        grid_id = get_mesh_id(
            latents.shape[-3] // self.patch_size[0],
            latents.shape[-2] // self.patch_size[1],
            latents.shape[-1] // self.patch_size[2],
            t=0,
            f_shift=frame_index,
            action=False,
        ).to(self.device)[None]
        return {
            "noisy_latents": latents,
            "timesteps": timesteps,
            "grid_id": grid_id,
            "text_emb": prompt_emb,
            "ego_cond": ego_cond,
        }

    def _make_action_input(self, actions, timesteps, prompt_emb, ego_cond, chunk_index):
        grid_id = get_mesh_id(
            actions.shape[-3],
            actions.shape[-2],
            actions.shape[-1],
            t=1,
            f_shift=chunk_index,
            action=True,
        ).to(self.device)[None]
        return {
            "noisy_latents": actions,
            "timesteps": timesteps,
            "grid_id": grid_id,
            "text_emb": prompt_emb,
            "ego_cond": ego_cond,
        }

    def _prepare_timesteps(self, scheduler, num_steps):
        scheduler.set_timesteps(num_steps)
        return F.pad(scheduler.timesteps, (0, 1), mode="constant", value=0)

    def _prepare_video_timesteps(self, num_steps, denoise_fraction=1.0):

        stop_sigma_raw = max(0.0, 1.0 - denoise_fraction)
        raw_sigmas = torch.linspace(1.0, stop_sigma_raw, num_steps + 1)

        shift = self.video_scheduler.shift
        shifted_sigmas = shift * raw_sigmas / (1 + (shift - 1) * raw_sigmas)

        self.video_scheduler.sigmas = shifted_sigmas
        self.video_scheduler.timesteps = (
            shifted_sigmas * self.video_scheduler.num_train_timesteps
        )

        return self.video_scheduler.timesteps

    # ------------------------------------------------------------------
    # Action denormalization (PhysicalAI quantile-based)
    # ------------------------------------------------------------------

    def actions_to_trajectory_with_reset(self, pred_actions, gt_actions):
        """
        Inverse quantile-normalized action tensor -> absolute xy trajectory
        with pose reset at chunk boundaries.

        After each chunk, the starting position is reset to the GT position,
        preventing error accumulation across chunks.
        """
        ch = list(self.config.used_action_channel_ids[:2])
        chunk_steps = self.action_chunk_steps

        pred_chunks = pred_actions[:, ch, 1:, :, 0].float()
        gt_chunks = gt_actions[:, ch, 1:, :, 0].float()

        q01 = self.q01.to(pred_chunks.device)
        q99 = self.q99.to(pred_chunks.device)
        pred_chunks = (pred_chunks + 1.0) / 2.0 * (q99 - q01 + 1e-6)[:, None, None] + q01[:, None, None]
        gt_chunks = (gt_chunks + 1.0) / 2.0 * (q99 - q01 + 1e-6)[:, None, None] + q01[:, None, None]

        num_chunks = pred_chunks.shape[2]
        gt_traj_full = torch.cumsum(gt_chunks.flatten(2), dim=-1)

        pred_positions = []
        gt_positions = []
        for k in range(num_chunks):
            if k == 0:
                start = torch.zeros(pred_chunks.shape[0], 2, 1, device=pred_chunks.device)
            else:
                start = gt_traj_full[:, :, k * chunk_steps - 1 : k * chunk_steps]
            pred_positions.append(start + torch.cumsum(pred_chunks[:, :, k], dim=-1))
            gt_positions.append(start + torch.cumsum(gt_chunks[:, :, k], dim=-1))

        pred_traj = torch.cat(pred_positions, dim=-1).cpu().numpy()
        gt_traj_np = torch.cat(gt_positions, dim=-1).cpu().numpy()
        return pred_traj, gt_traj_np

    # ------------------------------------------------------------------
    # Ego conditioning helper
    # ------------------------------------------------------------------

    def _gt_ego_cond(self, gt_ego_vel_acc, gt_ego_curv, num_frames):
        """Return GT ego conditioning for first `num_frames` chunks."""
        return {
            "velocity_acc": gt_ego_vel_acc[:, :num_frames].to(self.device, self.dtype),
            "curvature": gt_ego_curv[:, :num_frames].to(self.device, self.dtype),
        }

    # ------------------------------------------------------------------
    # Autoregressive rollout with GT updates
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_autoregressive(self, batch, num_video_steps=3, num_action_steps=10, seed=0,
                                   video_denoise_fraction=0.6):
        """
        Autoregressive rollout over the full clip with GT updates.

        At each step: predict video frame + action chunk, then register GT
        video and GT action into cache before predicting the next step.

        Returns:
            pred_actions: (1, action_dim, chunk_count, chunk_steps, 1)
            pred_latents: (1, C, T_lat, H', W')
        """
        images = batch["images"][:1].to(self.device)  # (1, T_video, 3, H, W)
        actions = batch["actions"][:1].to(self.device, self.dtype)
        actions_mask = batch["actions_mask"][:1].to(self.device)
        ego_vel_acc = batch["ego_velocity_acc"][:1]  # (1, num_ego_chunks, 4)
        ego_curv = batch["curvature"][:1]  # (1, num_ego_chunks)

        prompts_per_chunk = batch["prompts"]  # list of N_chunks elements (each a 1-elem list after collation)

        def _chunk_prompt_emb(ci):
            p = prompts_per_chunk[ci] if ci < len(prompts_per_chunk) else prompts_per_chunk[0]
            if isinstance(p, (list, tuple)):
                p = p[0]
            return self._encode_prompt([p])  # (1, L, D)

        cond_prompt_emb = _chunk_prompt_emb(0)

        # Encode all video frames to latents
        gt_latents = self._encode_images(images).to(self.dtype)
        T_lat = gt_latents.shape[2]
        _, C_lat, _, H_lat, W_lat = gt_latents.shape
        action_dim = actions.shape[1]
        action_steps_per_chunk = self.action_chunk_steps

        # Setup timesteps (video uses partial denoising via denoise_fraction)
        latent_timesteps = self._prepare_video_timesteps(num_video_steps, video_denoise_fraction)

        action_scheduler = FlowMatchScheduler(
            shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True,
        )
        action_scheduler.set_timesteps(num_action_steps)
        action_timesteps = F.pad(action_scheduler.timesteps, (0, 1), mode="constant", value=0)

        # Token counts for KV cache
        ph, pw = self.patch_size[1], self.patch_size[2]
        latent_tokens_per_chunk = (H_lat // ph) * (W_lat // pw)
        action_tokens_per_chunk = action_steps_per_chunk

        # Zero action condition
        action_cond = torch.zeros(
            1, action_dim, 1, action_steps_per_chunk, 1,
            device=self.device, dtype=self.dtype,
        )

        denoised_latents = []
        denoised_action_chunks = []

        rng_devices = self._rng_devices()

        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(seed)

            # Setup KV cache
            self.transformer.create_empty_cache(
                self.cache_name, self.config.attn_window,
                latent_tokens_per_chunk, action_tokens_per_chunk,
                self.device, self.dtype, batch_size=1,
            )

            try:
                # Register GT frame 0
                ego_cond_empty = self._gt_ego_cond(ego_vel_acc, ego_curv, 0)
                zero_t = torch.zeros((1, 1), device=self.device, dtype=torch.float32)

                input_dict = self._make_video_input(
                    gt_latents[:, :, 0:1], zero_t, cond_prompt_emb, ego_cond_empty, frame_index=0,
                )
                self.transformer(
                    input_dict,
                    update_cache=2, cache_name=self.cache_name, action_mode=False,
                )

                # Register action condition chunk (zeros)
                input_dict = self._make_action_input(
                    action_cond, zero_t, cond_prompt_emb, ego_cond_empty, chunk_index=0,
                )
                self.transformer(
                    input_dict,
                    update_cache=2, cache_name=self.cache_name, action_mode=True,
                )

                # Autoregressive loop
                for frame_idx in range(1, T_lat):
                    cur_ego_cond = self._gt_ego_cond(ego_vel_acc, ego_curv, frame_idx)
                    cur_prompt_emb = _chunk_prompt_emb(frame_idx)

                    # -- Denoise one video frame --
                    chunk_latents = torch.randn(
                        1, C_lat, 1, H_lat, W_lat,
                        device=self.device, dtype=self.dtype,
                    )
                    for i, t in enumerate(latent_timesteps):
                        last_step = (i == len(latent_timesteps) - 1)
                        timestep_tensor = torch.full(
                            (1, 1), float(t.item()),
                            device=self.device, dtype=torch.float32,
                        )
                        input_dict = self._make_video_input(
                            chunk_latents, timestep_tensor, cur_prompt_emb, cur_ego_cond,
                            frame_index=frame_idx,
                        )
                        pred = self.transformer(
                            input_dict,
                            update_cache=(1 if last_step else 0),
                            cache_name=self.cache_name,
                            action_mode=False,
                        )
                        if last_step:
                            continue
                        pred = data_seq_to_patch(
                            self.patch_size, pred, 1, H_lat, W_lat, batch_size=1,
                        )
                        chunk_latents = self.video_scheduler.step(pred, t, chunk_latents)
                    denoised_latents.append(chunk_latents)

                    # -- Denoise one action chunk --
                    action_mask_slice = actions_mask[:, :, frame_idx:frame_idx + 1, :, :].bool()
                    chunk_actions = torch.randn(
                        1, action_dim, 1, action_steps_per_chunk, 1,
                        device=self.device, dtype=self.dtype,
                    )
                    chunk_actions = torch.where(action_mask_slice, chunk_actions, torch.zeros_like(chunk_actions))

                    for i, t in enumerate(action_timesteps):
                        last_step = (i == len(action_timesteps) - 1)
                        timestep_tensor = torch.full(
                            (1, 1), float(t.item()),
                            device=self.device, dtype=torch.float32,
                        )
                        input_dict = self._make_action_input(
                            chunk_actions, timestep_tensor, cur_prompt_emb, cur_ego_cond,
                            chunk_index=frame_idx,
                        )
                        pred = self.transformer(
                            input_dict,
                            update_cache=(1 if last_step else 0),
                            cache_name=self.cache_name,
                            action_mode=True,
                        )
                        if last_step:
                            continue
                        pred = rearrange(pred, "b (f n) c -> b c f n 1", f=1)
                        chunk_actions = action_scheduler.step(pred, t, chunk_actions)
                        chunk_actions = torch.where(
                            action_mask_slice, chunk_actions, torch.zeros_like(chunk_actions),
                        )
                    denoised_action_chunks.append(chunk_actions)

                    # -- Clear predicted KVs so GT overwrites them --
                    self.transformer.clear_pred_cache(self.cache_name)

                    # -- Register GT video frame --
                    input_dict = self._make_video_input(
                        gt_latents[:, :, frame_idx:frame_idx + 1].to(self.dtype),
                        zero_t, cur_prompt_emb, cur_ego_cond, frame_index=frame_idx,
                    )
                    self.transformer(
                        input_dict,
                        update_cache=2, cache_name=self.cache_name, action_mode=False,
                    )

                    # -- Register GT action chunk --
                    gt_action_chunk = actions[:, :, frame_idx:frame_idx + 1, :, :]
                    input_dict = self._make_action_input(
                        gt_action_chunk, zero_t, cur_prompt_emb, cur_ego_cond, chunk_index=frame_idx,
                    )
                    self.transformer(
                        input_dict,
                        update_cache=2, cache_name=self.cache_name, action_mode=True,
                    )

            finally:
                self.transformer.clear_cache(self.cache_name)

        # Assemble final tensors
        pred_latents = torch.cat(
            [gt_latents[:, :, 0:1].to(self.dtype)] + denoised_latents, dim=2,
        )
        pred_actions = torch.cat([action_cond] + denoised_action_chunks, dim=2)

        return pred_actions, pred_latents

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_ade_fde(self, pred_actions, gt_actions):
        """
        ADE: mean displacement error over all steps.
        FDE: mean of per-chunk final-step displacement errors.
        """
        pred_traj, gt_traj = self.actions_to_trajectory_with_reset(pred_actions, gt_actions)
        diff = pred_traj[0] - gt_traj[0]
        l2 = np.sqrt((diff ** 2).sum(axis=0))
        chunk_steps = self.action_chunk_steps
        num_chunks = l2.shape[0] // chunk_steps
        chunk_end_indices = [(k + 1) * chunk_steps - 1 for k in range(num_chunks)]
        fde = float(np.mean(l2[chunk_end_indices]))
        return float(l2.mean()), fde


# ===========================================================================
# Multi-GPU worker & aggregation
# ===========================================================================


def _worker_fn(rank, args, result_dir):
    """Per-GPU evaluation worker.  Called by ``mp.spawn`` with *rank* in [0, num_gpus)."""
    device = f"cuda:{rank}"

    config = VA_CONFIGS[args.config_name]
    config.rank = rank
    config.local_rank = rank
    config.world_size = args.num_gpus
    config.split = "test"

    evaluator = PhysicalAIEvaluator(
        config,
        checkpoint_path=args.checkpoint,
        attn_mode="flashattn",
        device=device,
    )

    dataset = PhysicalAITestDataset(
        config=config,
        max_clips=args.num_eval_samples,
        clip_ids=getattr(args, "clip_ids_list", None),
        clip_prompts_path=getattr(args, "clip_prompts_path", None),
    )
    total = len(dataset)

    # Shard: rank handles indices rank, rank+num_gpus, rank+2*num_gpus, ...
    indices = list(range(rank, total, args.num_gpus))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    rank_results = []
    for local_idx, batch in enumerate(tqdm(loader, desc=f"GPU{rank}", position=rank, leave=True)):
        global_idx = indices[local_idx]

        pred_actions, pred_latents = evaluator.predict_autoregressive(
            batch,
            num_video_steps=args.num_video_steps,
            num_action_steps=args.num_action_steps,
            seed=args.seed,
            video_denoise_fraction=args.video_denoise_fraction,
        )

        gt_actions = batch["actions"][:1].to(evaluator.device)
        ade, fde = evaluator.compute_ade_fde(pred_actions, gt_actions)

        rank_results.append({
            "global_idx": global_idx,
            "token": batch["scene_token"][0],
            "ade": ade,
            "fde": fde,
        })

    # Persist this rank's results so the main process can aggregate.
    torch.save(rank_results, Path(result_dir) / f"rank_{rank}.pt")


def evaluate(args):
    # Load clip IDs from CSV if provided
    if getattr(args, "clip_ids_csv", None):
        import pandas as pd
        csv_df = pd.read_csv(args.clip_ids_csv)
        args.clip_ids_list = csv_df["clip_id"].tolist()
        logger.info(f"Loaded {len(args.clip_ids_list)} clip IDs from {args.clip_ids_csv}")
    else:
        args.clip_ids_list = None

    available_gpus = torch.cuda.device_count()
    if args.num_gpus > available_gpus:
        raise RuntimeError(f"Requested {args.num_gpus} GPUs but only {available_gpus} available.")

    result_dir = tempfile.mkdtemp(prefix="physicalai_eval_")
    logger.info(
        f"Spawning {args.num_gpus} GPU workers  "
        f"video_steps={args.num_video_steps}  action_steps={args.num_action_steps}  "
        f"video_denoise_fraction={args.video_denoise_fraction}"
    )

    mp.spawn(_worker_fn, args=(args, result_dir), nprocs=args.num_gpus, join=True)

    # Aggregate results from all ranks
    all_results = []
    for rank in range(args.num_gpus):
        rank_results = torch.load(Path(result_dir) / f"rank_{rank}.pt", weights_only=False)
        all_results.extend(rank_results)
    all_results.sort(key=lambda r: r["global_idx"])

    ade_list = [r["ade"] for r in all_results]
    fde_list = [r["fde"] for r in all_results]

    print(f"\n=== PhysicalAI Parallel Evaluation ({len(ade_list)} samples, {args.num_gpus} GPUs) ===")
    print(f"  video_steps={args.num_video_steps}  action_steps={args.num_action_steps}  "
          f"video_denoise_fraction={args.video_denoise_fraction}")
    if ade_list:
        print(f"  ADE : {np.mean(ade_list):.4f} m  (±{np.std(ade_list):.4f})")
        print(f"  FDE : {np.mean(fde_list):.4f} m  (±{np.std(fde_list):.4f})")
    else:
        print("  No valid samples evaluated.")


def main():
    parser = argparse.ArgumentParser(description="PhysicalAI rollout evaluation")
    parser.add_argument("--config-name", default="physicalai_cfg")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=8,
        help="Number of GPUs to use for parallel inference.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-eval-samples",
        type=int,
        default=1000,
        help="Maximum number of test clips to evaluate.",
    )
    parser.add_argument("--num-video-steps", type=int, default=3, help="Video denoising steps.")
    parser.add_argument("--num-action-steps", type=int, default=10, help="Action denoising steps.")
    parser.add_argument(
        "--video-denoise-fraction", type=float, default=0.6,
        help="Partial video denoising (0.0–1.0). "
             "1.0 = fully denoise to sigma=0 (default action-style).",
    )
    parser.add_argument(
        "--clip-prompts-path",
        type=str,
        default="DriveWAM/src/physicalai/eval_data/prompts_test_sample_1k.json",
        help="JSON mapping clip_id -> prompt. Empty/None falls back to config.prompt.",
    )
    parser.add_argument(
        "--clip-ids-csv",
        type=str,
        default="DriveWAM/src/physicalai/eval_data/test_sample_1k.csv",
        help="Path to a CSV file with a 'clip_id' column. If provided, uses these clip IDs "
             "instead of loading from the parquet split index.",
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    init_logger()
    main()
