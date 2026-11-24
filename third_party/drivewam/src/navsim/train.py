import argparse
import gc
import json
import os
import sys
from pathlib import Path

import swanlab as wandb
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import VA_CONFIGS
from dataset.navsim_dataset import NavSimEpisodeDataset
from distributed.fsdp import apply_ac, shard_model
from distributed.util import _configure_model, dist_max, dist_mean, init_distributed
from modules.navsim_model import WanTransformer3DNavSimModel
from modules.utils import WanVAEStreamingWrapper, load_text_encoder, load_tokenizer, load_vae
from utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    sample_timestep_id,
    warmup_constant_lambda,
)


class Trainer:
    def __init__(self, config):
        if config.enable_wandb and config.rank == 0:
            wandb.login(api_key=os.environ["WANDB_API_KEY"])
            self.wandb = wandb
            self.wandb.init(
                project=os.getenv("WANDB_PROJECT", "DriveWAM"),
                config=config,
                mode="local",
                name="navsim_train",
            )
            logger.info("WandB logging enabled")

        self.step = 0
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size
        self.prompt_embed_cache = {}

        logger.info("Loading NavSim model...")
        transformer_path = os.path.join(config.wan22_pretrained_model_name_or_path, "transformer")

        self.transformer = WanTransformer3DNavSimModel.from_pretrained(
            transformer_path,
            torch_dtype=torch.float32,
        ).to("cpu")

        logger.info("Setting up activation checkpointing...")
        apply_ac(self.transformer)

        logger.info("Setting up FSDP...")
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_model,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self.transformer.train()
        self.transformer.requires_grad_(True)

        logger.info("Loading VAE and text encoder...")
        self.vae = load_vae(
            os.path.join(config.wan22_pretrained_model_name_or_path, "vae"),
            torch_dtype=self.dtype,
            torch_device=self.device,
        )
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)

        self.tokenizer = load_tokenizer(
            os.path.join(config.wan22_pretrained_model_name_or_path, "tokenizer"),
        )
        self.text_encoder = load_text_encoder(
            os.path.join(config.wan22_pretrained_model_name_or_path, "text_encoder"),
            torch_dtype=self.dtype,
            torch_device=self.device,
        )
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(
            [p for p in self.transformer.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )
        lr_decay_milestones = sorted(getattr(config, "lr_decay_milestones", []))
        lr_decay_gamma = getattr(config, "lr_decay_gamma", 1.0)

        def lr_lambda(step):
            scale = warmup_constant_lambda(step, warmup_steps=config.warmup_steps)
            for milestone in lr_decay_milestones:
                if step >= milestone:
                    scale *= lr_decay_gamma
            return scale

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambda,
            last_epoch=-1,
        )

        logger.info("Setting up NavSim dataset...")
        train_dataset = NavSimEpisodeDataset(config=config)
        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=config.world_size,
                rank=config.rank,
                shuffle=True,
                seed=42,
            )
            if config.world_size > 1
            else None
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None),
            num_workers=config.load_worker,
            sampler=train_sampler,
        )

        self.train_scheduler_latent = FlowMatchScheduler(
            shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(
            shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self.train_scheduler_action.set_timesteps(1000, training=True)

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        self.train_loader_iter = None

    def _get_next_batch(self):
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)

        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(self.train_loader.sampler.epoch + 1)
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)

        return batch

    def _move_to_device(self, item):
        if torch.is_tensor(item):
            return item.to(self.device)
        if isinstance(item, dict):
            return {k: self._move_to_device(v) for k, v in item.items()}
        return item

    def normalize_latents(self, latents, latents_mean, latents_std):
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device, dtype=latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device, dtype=latents.dtype)
        return ((latents.float() - latents_mean) * latents_std).to(latents)

    @torch.no_grad()
    def _encode_images_to_latents(self, images):
        batch_size, num_frames = images.shape[:2]
        images = images.float()
        images = F.interpolate(
            images.flatten(0, 1),
            size=(self.config.height, self.config.width),
            mode="bilinear",
            align_corners=False,
        )
        images = rearrange(images, "(b t) c h w -> b t c h w", b=batch_size, t=num_frames)
        videos = rearrange(images / 255.0 * 2.0 - 1.0, "b t c h w -> b c t h w").to(self.dtype)

        posterior = self.vae.encode(videos).latent_dist
        mu = posterior.mean

        latents_mean = torch.tensor(self.vae.config.latents_mean, device=mu.device, dtype=mu.dtype)
        latents_std = torch.tensor(self.vae.config.latents_std, device=mu.device, dtype=mu.dtype)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        return mu_norm

    def _build_latent_prediction_mask(self, latents):
        latent_mask = torch.ones(
            latents.shape[0],
            1,
            latents.shape[2],
            1,
            1,
            dtype=torch.bool,
            device=latents.device,
        )
        latent_mask[:, :, 0] = False
        return latent_mask

    def _dense_mask_to_frame_mask(self, dense_mask):
        return dense_mask.bool().any(dim=1).any(dim=-1).any(dim=-1)

    def _timesteps_to_sigmas(self, scheduler, timesteps):
        flat_timesteps = timesteps.flatten()
        timestep_ids = torch.argmin(
            (scheduler.timesteps[:, None].to(flat_timesteps.device) - flat_timesteps[None]).abs(),
            dim=0,
        )
        return scheduler.sigmas.to(flat_timesteps.device)[timestep_ids].reshape_as(timesteps)

    @torch.no_grad()
    def _predict_clean_latents(self, input_dict, pred):
        latent_pred_seq = pred[0].detach()
        latent_pred = data_seq_to_patch(
            self.patch_size,
            latent_pred_seq,
            input_dict["latent_dict"]["targets"].shape[-3],
            input_dict["latent_dict"]["targets"].shape[-2],
            input_dict["latent_dict"]["targets"].shape[-1],
            batch_size=latent_pred_seq.shape[0],
        )

        noisy_latents = input_dict["latent_dict"]["noisy_latents"].detach()
        sigmas = self._timesteps_to_sigmas(
            self.train_scheduler_latent,
            input_dict["latent_dict"]["timesteps"],
        ).to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        pred_clean_latents = noisy_latents - sigmas[:, None, :, None, None] * latent_pred.to(noisy_latents.dtype)
        pred_clean_latents = torch.where(
            input_dict["latent_dict"]["latents_mask"].bool(),
            pred_clean_latents,
            input_dict["latent_dict"]["latent"],
        )
        return pred_clean_latents

    @torch.no_grad()
    def _predict_clean_actions(self, input_dict, pred):
        """Decode model action velocity prediction → clean (normalized) actions via flow matching."""
        action_pred_seq = pred[1].detach()
        F_dim = input_dict["action_dict"]["targets"].shape[-3]
        action_pred = rearrange(action_pred_seq, "b (f n) c -> b c f n 1", f=F_dim)

        noisy_actions = input_dict["action_dict"]["noisy_latents"].detach()
        sigmas = self._timesteps_to_sigmas(
            self.train_scheduler_action,
            input_dict["action_dict"]["timesteps"],
        ).to(device=noisy_actions.device, dtype=noisy_actions.dtype)

        pred_clean = noisy_actions - sigmas[:, None, :, None, None] * action_pred.to(noisy_actions.dtype)
        pred_clean = pred_clean * input_dict["action_dict"]["actions_mask"].float()
        return pred_clean

    @torch.no_grad()
    def _build_joint_sampling_input(
        self,
        latent_noisy_latents,
        action_noisy_latents,
        latent_condition_latents,
        action_condition_latents,
        latent_timesteps,
        action_timesteps,
        latent_grid_id,
        action_grid_id,
        prompt_emb,
        ego_cond,
        action_mask,
        latent_cond,
        action_cond,
    ):
        latent_noisy_latents = latent_noisy_latents.clone()
        action_noisy_latents = action_noisy_latents.clone()
        latent_noisy_latents[:, :, 0:1] = latent_cond
        action_noisy_latents = torch.where(action_mask, action_noisy_latents, torch.zeros_like(action_noisy_latents))
        action_noisy_latents[:, :, 0:1] = action_cond

        return {
            "latent_dict": {
                "noisy_latents": latent_noisy_latents,
                "latent": latent_condition_latents,
                "timesteps": latent_timesteps,
                "cond_timesteps": torch.zeros_like(latent_timesteps),
                "grid_id": latent_grid_id,
                "text_emb": prompt_emb,
                "ego_cond": ego_cond,
            },
            "action_dict": {
                "noisy_latents": action_noisy_latents,
                "latent": action_condition_latents,
                "timesteps": action_timesteps,
                "cond_timesteps": torch.zeros_like(action_timesteps),
                "grid_id": action_grid_id,
                "text_emb": prompt_emb,
                "actions_mask": action_mask,
            },
            "chunk_size": self.config.frame_chunk_size,
            "window_size": self.config.attn_window,
        }

    @torch.no_grad()
    def _get_t5_prompt_embeds(self, prompt):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.config.text_max_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(
            text_input_ids.to(text_encoder_device),
            mask.to(text_encoder_device),
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat(
                    [u, u.new_zeros(self.config.text_max_length - u.size(0), u.size(1))]
                )
                for u in prompt_embeds
            ],
            dim=0,
        )
        return prompt_embeds.view(batch_size, self.config.text_max_length, -1)

    @torch.no_grad()
    def _encode_prompts(self, prompts):
        if self.config.enable_text_encoder_cache:
            embeds = []
            for prompt in prompts:
                if prompt not in self.prompt_embed_cache:
                    self.prompt_embed_cache[prompt] = self._get_t5_prompt_embeds(prompt)[0]
                embeds.append(self.prompt_embed_cache[prompt])
            return torch.stack(embeds, dim=0)
        return self._get_t5_prompt_embeds(prompts)

    @torch.no_grad()
    def _add_noise(
        self,
        latent,
        train_scheduler,
        action_mask=None,
        action_mode=False,
        noisy_cond_prob=0.0,
        noise_mask=None,
    ):
        B, C, F_, H, W = latent.shape
        clean_latent = latent

        # Sample timesteps independently for each batch sample: (B, F_)
        timestep_ids = sample_timestep_id(batch_size=B * F_, num_train_timesteps=train_scheduler.num_train_timesteps)
        timestep_ids = timestep_ids.view(B, F_)
        noise = torch.zeros_like(latent).normal_()

        # Process each batch sample with independent timesteps
        noisy_latents_list = []
        targets_list = []
        timesteps_list = []
        for b in range(B):
            timesteps_b = train_scheduler.timesteps[timestep_ids[b]].to(device=self.device)
            noisy_latents_b = train_scheduler.add_noise(latent[b:b+1], noise[b:b+1], timesteps_b, t_dim=2)
            targets_b = train_scheduler.training_target(latent[b:b+1], noise[b:b+1], timesteps_b)
            noisy_latents_list.append(noisy_latents_b)
            targets_list.append(targets_b)
            timesteps_list.append(timesteps_b)

        noisy_latents = torch.cat(noisy_latents_list, dim=0)
        targets = torch.cat(targets_list, dim=0)
        timesteps = torch.stack(timesteps_list, dim=0)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1

        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,
            latent.shape[-2] // patch_h,
            latent.shape[-1] // patch_w,
            t=1 if action_mode else 0,
            f_w=1,
            f_shift=0,
            action=action_mode,
        ).to(self.device)
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        # Decide noisy-cond per sample instead of one batch-wide coin flip.
        cond_timesteps = torch.zeros_like(timesteps)
        if noisy_cond_prob > 0:
            cond_sample_mask = torch.rand(B) < noisy_cond_prob
            latent_list = []
            for b in range(B):
                if bool(cond_sample_mask[b]):
                    # # each selected sample independently samples F_ condition timesteps
                    cond_timestep_ids = sample_timestep_id(
                        batch_size=F_,
                        min_timestep_bd=0.5,
                        max_timestep_bd=1.0,
                        num_train_timesteps=train_scheduler.num_train_timesteps,
                    )
                    cond_timesteps_b = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
                    cond_noise_b = torch.zeros_like(latent[b:b+1]).normal_()
                    latent_b = train_scheduler.add_noise(latent[b:b+1], cond_noise_b, cond_timesteps_b, t_dim=2)
                    latent_list.append(latent_b)
                    cond_timesteps[b] = cond_timesteps_b
                else:
                    latent_list.append(latent[b:b+1])
            latent = torch.cat(latent_list, dim=0)

        if noise_mask is not None:
            noise_frame_mask = self._dense_mask_to_frame_mask(noise_mask)
            noise_mask = noise_frame_mask[:, None, :, None, None]
            noisy_latents = torch.where(noise_mask, noisy_latents, clean_latent)
            targets = targets * noise_mask.float()
            latent = torch.where(noise_mask, latent, clean_latent)
            timesteps = torch.where(noise_frame_mask, timesteps, torch.zeros_like(timesteps))
            cond_timesteps = torch.where(noise_frame_mask, cond_timesteps, torch.zeros_like(cond_timesteps))

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()
            action_frame_mask = self._dense_mask_to_frame_mask(action_mask)
            timesteps = torch.where(action_frame_mask, timesteps, torch.zeros_like(timesteps))
            cond_timesteps = torch.where(action_frame_mask, cond_timesteps, torch.zeros_like(cond_timesteps))

        return dict(
            timesteps=timesteps,
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps,
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        latents = self._encode_images_to_latents(batch_dict["images"])
        latent_mask = self._build_latent_prediction_mask(latents)
        text_emb = self._encode_prompts(batch_dict["prompt"])

        latent_dict = self._add_noise(
            latent=latents,
            train_scheduler=self.train_scheduler_latent,
            action_mask=None,
            action_mode=False,
            noisy_cond_prob=0.5,
            noise_mask=latent_mask,
        )
        action_dict = self._add_noise(
            latent=batch_dict["actions"],
            train_scheduler=self.train_scheduler_action,
            action_mask=batch_dict["actions_mask"],
            action_mode=True,
            noisy_cond_prob=0.0,
        )

        latent_dict["text_emb"] = text_emb
        latent_dict["latents_mask"] = latent_mask
        latent_dict["ego_cond"] = {
            "velocity_acc": batch_dict["ego_velocity_acc"],
            "history_poses": batch_dict["history_poses"],
            "command_ids": batch_dict["command_id"],
        }
        action_dict["text_emb"] = text_emb
        action_dict["actions_mask"] = batch_dict["actions_mask"]

        return {
            "latent_dict": latent_dict,
            "action_dict": action_dict,
            "chunk_size": self.config.frame_chunk_size,
            "window_size": self.config.attn_window,
        }

    def compute_loss(self, input_dict, pred):
        latent_pred, action_pred = pred
        action_pred = rearrange(
            action_pred,
            "b (f n) c -> b c f n 1",
            f=input_dict["action_dict"]["targets"].shape[-3],
        )
        latent_pred = data_seq_to_patch(
            self.patch_size,
            latent_pred,
            input_dict["latent_dict"]["targets"].shape[-3],
            input_dict["latent_dict"]["targets"].shape[-2],
            input_dict["latent_dict"]["targets"].shape[-1],
            batch_size=latent_pred.shape[0],
        )
        Bn, Fn = input_dict["latent_dict"]["timesteps"].shape
        Ba, Fa = input_dict["action_dict"]["timesteps"].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(
            input_dict["latent_dict"]["timesteps"].flatten()
        ).reshape(Bn, Fn)
        action_loss_weight = self.train_scheduler_action.training_weight(
            input_dict["action_dict"]["timesteps"].flatten()
        ).reshape(Ba, Fa)

        latent_loss = F.mse_loss(
            latent_pred.float(),
            input_dict["latent_dict"]["targets"].float().detach(),
            reduction="none",
        )
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        latent_mask = input_dict["latent_dict"]["latents_mask"].float().expand_as(
            input_dict["latent_dict"]["targets"]
        )
        latent_loss = latent_loss * latent_mask
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        latent_mask = latent_mask.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        latent_loss_per_frame = latent_loss.sum(dim=1)
        latent_mask_per_frame = latent_mask.sum(dim=1)
        latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()

        action_loss = F.mse_loss(
            action_pred.float(),
            input_dict["action_dict"]["targets"].float().detach(),
            reduction="none",
        )
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * input_dict["action_dict"]["actions_mask"].float()
        action_loss = action_loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        action_mask = (
            input_dict["action_dict"]["actions_mask"].float().permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        )
        action_loss_per_frame = action_loss.sum(dim=1)
        action_mask_per_frame = action_mask.sum(dim=1)
        action_loss = (action_loss_per_frame / (action_mask_per_frame + 1e-6)).mean()

        return latent_loss / self.gradient_accumulation_steps, action_loss / self.gradient_accumulation_steps

    def _train_step(self, batch, batch_idx):
        batch = self._move_to_device(batch)
        input_dict = self._prepare_input_dict(batch)

        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        if not should_sync:
            self.transformer.set_requires_gradient_sync(False)
        else:
            self.transformer.set_requires_gradient_sync(True)

        output = self.transformer(input_dict, train_mode=True)
        latent_loss, action_loss = self.compute_loss(input_dict, output)
        loss = latent_loss + action_loss
        loss.backward()

        losses = {"latent_loss": latent_loss.detach(), "action_loss": action_loss.detach()}
        if should_sync:
            total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            losses["total_norm"] = total_norm
            losses["should_log"] = True
        else:
            losses["should_log"] = False
        return losses

    def save_checkpoint(self):
        try:
            state_dict = get_model_state_dict(
                self.transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}

            if self.config.rank == 0:
                checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                transformer_dir = checkpoint_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)
                model_file = transformer_dir / "diffusion_pytorch_model.safetensors"
                save_file(state_dict_bf16, model_file)

                config_file = transformer_dir / "config.json"
                config_dict = dict(self.transformer.config)
                config_dict.pop("_name_or_path", None)
                config_dict["_class_name"] = "WanTransformer3DNavSimModel"
                with open(config_file, "w") as f:
                    json.dump(config_dict, f, indent=2)

                logger.info(f"Checkpoint saved successfully at step {self.step}")

            if dist.is_initialized():
                dist.barrier()
        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save checkpoint: {e}")
                import traceback

                logger.error(traceback.format_exc())
            if dist.is_initialized():
                dist.barrier()

    def train(self):
        logger.info(f"Starting NavSim training for {self.config.num_steps} steps...")
        self.transformer.train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step,
        )

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            batch = self._get_next_batch()
            losses = self._train_step(batch, step_in_accumulation)

            accumulated_latent_losses.append(losses["latent_loss"])
            accumulated_action_losses.append(losses["action_loss"])
            step_in_accumulation += 1

            if losses["should_log"]:
                lr = self.lr_scheduler.get_last_lr()[0]
                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()

                accumulated_latent_losses = []
                accumulated_action_losses = []
                step_in_accumulation = 0

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    total_norm = losses["total_norm"]
                    progress_bar.n += self.gradient_accumulation_steps
                    progress_bar.set_postfix(
                        {
                            "latent_loss": f"{latent_loss_show:.4f}",
                            "action_loss": f"{action_loss_show:.4f}",
                            "step": self.step,
                            "grad_norm": f"{total_norm.item():.2f}",
                            "lr": f"{lr:.2e}",
                        }
                    )
                    if self.config.enable_wandb:
                        log_dict = {
                            "loss_metrics/global_avg_video_loss": latent_loss_show,
                            "loss_metrics/global_avg_action_loss": action_loss_show,
                            "loss_metrics/global_max_video_loss": max_latent_loss_show,
                            "loss_metrics/global_max_action_loss": max_action_loss_show,
                            "grad_norm": total_norm.item(),
                            "lr": lr,
                        }
                        self.wandb.log(log_dict, step=self.step)

                self.step += 1
                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    self.save_checkpoint()

            if dist.is_initialized():
                dist.barrier()

        progress_bar.close()
        logger.info("NavSim training completed!")


def run(args):
    config = VA_CONFIGS[args.config_name]
    config.dataset_path = args.dataset_path

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    machine_rank = int(os.environ.get("DISTRIBUTED_NODE_RANK", 0))
    machine_num = int(os.environ.get("DISTRIBUTED_NODE_COUNT", 1))
    master_addr = os.environ.get("DISTRIBUTED_MASTER_HOSTS", "127.0.0.1")
    tcp_port = os.environ.get("LUBAN_AVAILBLE_PORT_0", None)
    single_machine = machine_num == 1

    if single_machine:
        init_distributed(world_size, local_rank, rank)
        print(f"Initialized distributed training with local rank {local_rank}")
    else:
        dist_url = f"tcp://{master_addr}:{tcp_port}"
        num_gpus = torch.cuda.device_count()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            rank=local_rank + (machine_rank * num_gpus),
            world_size=num_gpus * machine_num,
        )
        rank = local_rank + (machine_rank * num_gpus)
        world_size = num_gpus * machine_num
        print(
            f"Initialized distributed training with local rank {local_rank} global rank {rank}, dist_url {dist_url}"
        )

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")

    trainer = Trainer(config)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Train DriveWAM NavSim model")
    parser.add_argument("--config-name", type=str, default="navsim_cfg")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./data/navsim/trainval",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
