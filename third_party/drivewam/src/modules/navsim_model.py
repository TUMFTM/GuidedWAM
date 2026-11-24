import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange

from .model import WanAttention, WanTransformer3DModel, WanRotaryPosEmbed, WanTimeTextImageEmbedding, FlexAttnFunc

__all__ = ["WanTransformer3DNavSimModel"]

class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        cross_attn_norm=False,
        eps=1e-6,
        attn_mode: str = "flashattn",
    ):
        super().__init__()

        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            attn_mode=attn_mode,
        )

        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
            attn_mode=attn_mode,
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # Ego condition length is tiny, so use standard attention and avoid Flex cross-mask assumptions.
        self.attn3 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
            attn_mode="torch",
        )
        self.norm4 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states,
        text_hidden_states,
        ego_hidden_states,
        temb,
        rotary_emb,
        update_cache=0,
        cache_name="pos",
        batch_size=None,
        tokens_per_batch=None,
        component_lens=None,
    ):
        temb_scale_shift_table = self.scale_shift_table[None] + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = rearrange(
            temb_scale_shift_table, "b l n c -> b n l c"
        ).chunk(6, dim=1)
        shift_msa = shift_msa.squeeze(1)
        scale_msa = scale_msa.squeeze(1)
        gate_msa = gate_msa.squeeze(1)
        c_shift_msa = c_shift_msa.squeeze(1)
        c_scale_msa = c_scale_msa.squeeze(1)
        c_gate_msa = c_gate_msa.squeeze(1)

        norm_hidden_states = (self.norm1(hidden_states.float()) * (1.0 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            norm_hidden_states,
            norm_hidden_states,
            rotary_emb,
            update_cache=update_cache,
            cache_name=cache_name,
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        text_attn_output = self.attn2(
            norm_hidden_states,
            text_hidden_states,
            text_hidden_states,
            None,
            update_cache=0,
            cache_name=cache_name,
        )
        hidden_states = hidden_states + text_attn_output

        norm_hidden_states = self.norm4(hidden_states.float()).type_as(hidden_states)
        if norm_hidden_states.shape[0] == ego_hidden_states.shape[0]:
            ego_attn_output = self.attn3(
                norm_hidden_states,
                ego_hidden_states,
                ego_hidden_states,
                None,
                update_cache=0,
                cache_name=cache_name,
            )
        else:
            if batch_size is None or tokens_per_batch is None:
                raise ValueError("batch_size and tokens_per_batch are required for flattened ego cross-attention")

            active_token_count = batch_size * tokens_per_batch

            # Packed sequence is component-major: [compA: s0..s_{B-1}][compB: ...]...
            # A plain "1 (b s) c -> b s c" assumes sample-major, so when B>1 one sample's ego gets applied to another sample's tokens (train/inference mismatch).
            # Fix: split per component -> (B, comp_len, C) -> concat into true (B, tokens_per_batch, C) for attention -> scatter back to component-major order.
            if component_lens is None:
                raise ValueError("component_lens is required for flattened ego cross-attention")
            comp_lens = list(component_lens)
            comp_blocks = torch.split(
                norm_hidden_states[:, :active_token_count],
                [cl * batch_size for cl in comp_lens],
                dim=1,
            )
            ego_query = torch.cat(
                [
                    rearrange(cb, "1 (b s) c -> b s c", b=batch_size, s=cl)
                    for cb, cl in zip(comp_blocks, comp_lens)
                ],
                dim=1,
            )  # (B, tokens_per_batch, C): row b holds all tokens of sample b
            ego_attn_output = self.attn3(
                ego_query,
                ego_hidden_states,
                ego_hidden_states,
                None,
                update_cache=0,
                cache_name=cache_name,
            )
            ego_out_blocks = torch.split(ego_attn_output, comp_lens, dim=1)
            ego_attn_output = torch.cat(
                [rearrange(ob, "b s c -> 1 (b s) c") for ob in ego_out_blocks],
                dim=1,
            )  # scatter back to component-major order

            if hidden_states.shape[1] > active_token_count:
                pad_tokens = hidden_states.shape[1] - active_token_count
                ego_attn_output = torch.cat(
                    [
                        ego_attn_output,
                        ego_attn_output.new_zeros(1, pad_tokens, ego_attn_output.shape[-1]),
                    ],
                    dim=1,
                )

        hidden_states = hidden_states + ego_attn_output

        norm_hidden_states = (self.norm3(hidden_states.float()) * (1.0 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states


class WanTransformer3DNavSimModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
        "patch_embedding_mlp",
        "condition_embedder",
        "condition_embedder_action",
        "ego_velocity_acc_mlp",
        "ego_history_mlp",
        "norm",
    ]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm4",
        "norm3",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size=[1, 2, 2],
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=48,
        out_channels=48,
        action_dim=30,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=14336,
        num_layers=30,
        cross_attn_norm=True,
        eps=1e-06,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        attn_mode="torch",
        ego_command_classes=4,
        ego_velocity_acc_dim=4,
        ego_history_pose_dim=12,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding_mlp = nn.Linear(
            in_channels * patch_size[0] * patch_size[1] * patch_size[2],
            inner_dim,
        )
        self.action_embedder = nn.Linear(action_dim, inner_dim)
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        self.condition_embedder_action = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        self.ego_velocity_acc_mlp = nn.Sequential(
            nn.Linear(ego_velocity_acc_dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, inner_dim),
        )
        self.ego_history_mlp = nn.Sequential(
            nn.Linear(ego_history_pose_dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, inner_dim),
        )
        self.ego_command_embed = nn.Embedding(ego_command_classes, inner_dim)

        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    cross_attn_norm,
                    eps,
                    attn_mode=attn_mode,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.action_proj_out = nn.Linear(inner_dim, action_dim)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        kwargs["output_loading_info"] = True
        model, loading_info = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        missing_keys = loading_info.get("missing_keys", [])
        unexpected_keys = loading_info.get("unexpected_keys", [])
        model._init_missing_navsim_parameters(missing_keys)
        if unexpected_keys:
            model._unexpected_keys = unexpected_keys
        if missing_keys:
            model._missing_keys = missing_keys
        return model

    def _get_materialization_device(self):
        for param in self.parameters():
            if not param.is_meta:
                return param.device
        return torch.device("cpu")

    @staticmethod
    def _materialize_module_if_needed(module, device):
        has_meta_param = any(param.is_meta for param in module.parameters(recurse=True))
        has_meta_buffer = any(buffer.is_meta for buffer in module.buffers(recurse=True))
        if has_meta_param or has_meta_buffer:
            module.to_empty(device=device)

    def _init_missing_navsim_parameters(self, missing_keys):
        needs_navsim_init = any(
            key.startswith(("ego_velocity_acc_mlp.", "ego_history_mlp.", "ego_command_embed."))
            or ".attn3." in key
            or ".norm4." in key
            for key in missing_keys
        )

        if not needs_navsim_init:
            return

        device = self._get_materialization_device()

        self._materialize_module_if_needed(self.ego_velocity_acc_mlp, device)
        self._materialize_module_if_needed(self.ego_history_mlp, device)
        self._materialize_module_if_needed(self.ego_command_embed, device)

        for module in [self.ego_velocity_acc_mlp, self.ego_history_mlp]:
            for submodule in module:
                if isinstance(submodule, nn.Linear):
                    nn.init.xavier_uniform_(submodule.weight)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)

        nn.init.normal_(self.ego_command_embed.weight, mean=0.0, std=0.02)

        for block in self.blocks:
            if isinstance(block, WanTransformerBlock):
                self._materialize_module_if_needed(block.attn3, device)
                self._materialize_module_if_needed(block.norm4, device)
                for linear in [
                    block.attn3.to_q,
                    block.attn3.to_k,
                    block.attn3.to_v,
                    block.attn3.to_out[0],
                ]:
                    nn.init.xavier_uniform_(linear.weight)
                    if linear.bias is not None:
                        nn.init.zeros_(linear.bias)
                if block.attn3.norm_q.weight is not None:
                    nn.init.ones_(block.attn3.norm_q.weight)
                if block.attn3.norm_k.weight is not None:
                    nn.init.ones_(block.attn3.norm_k.weight)
                if isinstance(block.norm4, FP32LayerNorm):
                    if block.norm4.weight is not None:
                        nn.init.ones_(block.norm4.weight)
                    if block.norm4.bias is not None:
                        nn.init.zeros_(block.norm4.bias)

    def clear_cache(self, cache_name):
        for block in self.blocks:
            block.attn1.clear_cache(cache_name)

    def clear_pred_cache(self, cache_name):
        for block in self.blocks:
            block.attn1.clear_pred_cache(cache_name)

    def create_empty_cache(self, cache_name, attn_window, latent_token_per_chunk, action_token_per_chunk, device, dtype, batch_size):
        total_tolen = (attn_window // 2) * latent_token_per_chunk + (attn_window // 2) * action_token_per_chunk
        for block in self.blocks:
            block.attn1.init_kv_cache(
                cache_name,
                total_tolen,
                self.num_attention_heads,
                self.attention_head_dim,
                device,
                dtype,
                batch_size,
            )

    def _input_embed(self, latents, input_type="latent"):
        if input_type == "latent":
            hidden_states = rearrange(
                latents,
                "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2],
            )
            hidden_states = self.patch_embedding_mlp(hidden_states)
        elif input_type == "action":
            hidden_states = rearrange(latents, "b c f h w -> b (f h w) c")
            hidden_states = self.action_embedder(hidden_states)
        elif input_type == "text":
            hidden_states = self.condition_embedder.text_embedder(latents)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
        return hidden_states

    def _get_ego_hidden_states(self, ego_cond, dtype):
        velocity_acc = ego_cond["velocity_acc"].to(dtype=self.ego_velocity_acc_mlp[0].weight.dtype)
        history_poses = ego_cond["history_poses"].flatten(1).to(dtype=self.ego_history_mlp[0].weight.dtype)
        target_history_dim = self.ego_history_mlp[0].in_features
        if history_poses.shape[-1] < target_history_dim:
            pad = target_history_dim - history_poses.shape[-1]
            history_poses = torch.nn.functional.pad(history_poses, (0, pad))
        elif history_poses.shape[-1] > target_history_dim:
            history_poses = history_poses[..., -target_history_dim:]

        command_ids = ego_cond["command_ids"].long()
        command_ids = torch.clamp(command_ids, min=0, max=self.ego_command_embed.num_embeddings - 1)

        ego_hidden_states = torch.stack(
            [
                self.ego_velocity_acc_mlp(velocity_acc),
                self.ego_history_mlp(history_poses),
                self.ego_command_embed(command_ids),
            ],
            dim=1,
        )
        return ego_hidden_states.to(dtype=dtype)

    def _time_embed(self, timesteps, H, W, dtype, action_mode=False):
        pach_scale_h, pach_scale_w = (1, 1) if action_mode else (self.patch_size[1], self.patch_size[2])
        latent_time_steps = torch.repeat_interleave(
            timesteps,
            (H // pach_scale_h) * (W // pach_scale_w),
            dim=1,
        )
        current_condition_embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        temb, timestep_proj = current_condition_embedder(latent_time_steps, dtype=dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
        return temb, timestep_proj

    def forward_train(self, input_dict):
        input_dict["latent_dict"]["noisy_latents"] = input_dict["latent_dict"]["noisy_latents"].to(torch.bfloat16)
        input_dict["latent_dict"]["latent"] = input_dict["latent_dict"]["latent"].to(torch.bfloat16)
        input_dict["action_dict"]["noisy_latents"] = input_dict["action_dict"]["noisy_latents"].to(torch.bfloat16)
        input_dict["action_dict"]["latent"] = input_dict["action_dict"]["latent"].to(torch.bfloat16)

        latent_dict = input_dict["latent_dict"]
        action_dict = input_dict["action_dict"]
        batch_size = latent_dict["noisy_latents"].shape[0]

        latent_hidden_states = self._input_embed(latent_dict["noisy_latents"], input_type="latent")
        action_hidden_states = self._input_embed(action_dict["noisy_latents"], input_type="action")
        text_hidden_states = self._input_embed(latent_dict["text_emb"], input_type="text")
        ego_hidden_states = self._get_ego_hidden_states(latent_dict["ego_cond"], dtype=latent_hidden_states.dtype)

        condition_latent_hidden_states = self._input_embed(latent_dict["latent"], input_type="latent")
        condition_action_hidden_states = self._input_embed(action_dict["latent"], input_type="action")

        # [noisy_latent, condition_latent, noisy_action, condition_action]
        component_lens = [
            latent_hidden_states.shape[1],
            condition_latent_hidden_states.shape[1],
            action_hidden_states.shape[1],
            condition_action_hidden_states.shape[1],
        ]
        tokens_per_batch = sum(component_lens)

        latent_hidden_states = latent_hidden_states.flatten(0, 1)[None]
        action_hidden_states = action_hidden_states.flatten(0, 1)[None]
        text_hidden_states = text_hidden_states.flatten(0, 1)[None]
        condition_latent_hidden_states = condition_latent_hidden_states.flatten(0, 1)[None]
        condition_action_hidden_states = condition_action_hidden_states.flatten(0, 1)[None]

        hidden_states = torch.cat(
            [
                latent_hidden_states,
                condition_latent_hidden_states,
                action_hidden_states,
                condition_action_hidden_states,
            ],
            dim=1,
        )

        latent_grid_id = latent_dict["grid_id"].permute(1, 0, 2).flatten(1)[None]
        action_grid_id = action_dict["grid_id"].permute(1, 0, 2).flatten(1)[None]
        full_grid_id = torch.cat([latent_grid_id] * 2 + [action_grid_id] * 2, dim=2)
        rotary_emb = self.rope(full_grid_id)[:, :, None]

        latent_time_steps = torch.cat([latent_dict["timesteps"].flatten(0, 1), latent_dict["cond_timesteps"].flatten(0, 1)])[None]
        action_time_steps = torch.cat([action_dict["timesteps"].flatten(0, 1), action_dict["cond_timesteps"].flatten(0, 1)])[None]
        latent_temb, latent_timestep_proj = self._time_embed(
            latent_time_steps,
            latent_dict["noisy_latents"].shape[-2],
            latent_dict["noisy_latents"].shape[-1],
            dtype=hidden_states.dtype,
            action_mode=False,
        )
        action_temb, action_timestep_proj = self._time_embed(
            action_time_steps,
            action_dict["noisy_latents"].shape[-2],
            action_dict["noisy_latents"].shape[-1],
            dtype=hidden_states.dtype,
            action_mode=True,
        )
        temb = torch.cat([latent_temb, action_temb], dim=1)
        timestep_proj = torch.cat([latent_timestep_proj, action_timestep_proj], dim=1)

        total_length = hidden_states.shape[1]
        padded_length = (128 - total_length % 128) % 128
        hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, padded_length))
        rotary_emb = torch.nn.functional.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
        temb = torch.nn.functional.pad(temb, (0, 0, 0, padded_length))
        timestep_proj = torch.nn.functional.pad(timestep_proj, (0, 0, 0, 0, 0, padded_length))

        split_list = [
            latent_hidden_states.shape[1],
            condition_latent_hidden_states.shape[1],
            action_hidden_states.shape[1],
            condition_action_hidden_states.shape[1],
            padded_length,
        ]

        FlexAttnFunc.init_mask(
            latent_dict["noisy_latents"].shape,
            action_dict["noisy_latents"].shape,
            padded_length,
            input_dict["chunk_size"],
            window_size=input_dict["window_size"],
            patch_size=self.patch_size,
            device=hidden_states.device,
        )

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                text_hidden_states,
                ego_hidden_states,
                timestep_proj,
                rotary_emb,
                update_cache=False,
                batch_size=batch_size,
                tokens_per_batch=tokens_per_batch,
                component_lens=component_lens,
            )

        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(2, dim=1)
        shift = shift.to(hidden_states.device).squeeze(1)
        scale = scale.to(hidden_states.device).squeeze(1)
        hidden_states = (self.norm_out(hidden_states.float()) * (1.0 + scale) + shift).type_as(hidden_states)
        latent_hidden_states, _, action_hidden_states, _, _ = torch.split(hidden_states, split_list, dim=1)
        latent_hidden_states = self.proj_out(latent_hidden_states)
        latent_hidden_states = rearrange(
            latent_hidden_states,
            "1 (b l) (n c) -> b (l n) c",
            n=math.prod(self.patch_size),
            b=batch_size,
        )
        action_hidden_states = self.action_proj_out(action_hidden_states)
        action_hidden_states = rearrange(action_hidden_states, "1 (b l) c -> b l c", b=batch_size)
        return latent_hidden_states, action_hidden_states

    def forward(
        self,
        input_dict,
        update_cache=0,
        cache_name="pos",
        action_mode=False,
        train_mode=False,
    ):
        if train_mode:
            return self.forward_train(input_dict)

        if action_mode:
            latent_hidden_states = rearrange(input_dict["noisy_latents"], "b c f h w -> b (f h w) c")
            latent_hidden_states = self.action_embedder(latent_hidden_states)
        else:
            latent_hidden_states = rearrange(
                input_dict["noisy_latents"],
                "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2],
            )
            latent_hidden_states = self.patch_embedding_mlp(latent_hidden_states)

        text_hidden_states = self.condition_embedder.text_embedder(input_dict["text_emb"])
        ego_hidden_states = self._get_ego_hidden_states(input_dict["ego_cond"], dtype=latent_hidden_states.dtype)

        latent_grid_id = input_dict["grid_id"]
        rotary_emb = self.rope(latent_grid_id)[:, :, None]
        pach_scale_h, pach_scale_w = (1, 1) if action_mode else (self.patch_size[1], self.patch_size[2])
        latent_time_steps = torch.repeat_interleave(
            input_dict["timesteps"],
            (input_dict["noisy_latents"].shape[-2] // pach_scale_h) * (input_dict["noisy_latents"].shape[-1] // pach_scale_w),
            dim=1,
        )
        current_condition_embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        temb, timestep_proj = current_condition_embedder(latent_time_steps, dtype=latent_hidden_states.dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))

        for block in self.blocks:
            latent_hidden_states = block(
                latent_hidden_states,
                text_hidden_states,
                ego_hidden_states,
                timestep_proj,
                rotary_emb,
                update_cache=update_cache,
                cache_name=cache_name,
            )

        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(2, dim=1)
        shift = shift.to(latent_hidden_states.device).squeeze(1)
        scale = scale.to(latent_hidden_states.device).squeeze(1)
        latent_hidden_states = (self.norm_out(latent_hidden_states.float()) * (1.0 + scale) + shift).type_as(
            latent_hidden_states
        )

        if action_mode:
            latent_hidden_states = self.action_proj_out(latent_hidden_states)
        else:
            latent_hidden_states = self.proj_out(latent_hidden_states)
            latent_hidden_states = rearrange(
                latent_hidden_states,
                "b l (n c) -> b (l n) c",
                n=math.prod(self.patch_size),
            )
        return latent_hidden_states
