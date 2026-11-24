import math
from typing import Callable, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from torch.nn.attention.flex_attention import BlockMask

from .model import WanAttention, WanAttentionSelectiveKV, WanTransformer3DModel, WanRotaryPosEmbed, WanTimeTextImageEmbedding, FlexAttnFunc, custom_sdpa

__all__ = ["WanTransformer3DPhysicalAIModel", "WanTransformer3DPhysicalAIModelSelectiveKV"]

class _SDPAModule(nn.Module):
    """Thin nn.Module wrapper around custom_sdpa so it can replace FlexAttn submodules."""
    def forward(self, q, k, v, **kwargs):
        return custom_sdpa(q, k, v)

_sdpa_for_inference = _SDPAModule()


def _set_attn_op(module, op):

    module._modules.pop('attn_op', None)
    if isinstance(op, nn.Module):
        module.attn_op = op          # register as submodule
    else:
        object.__setattr__(module, 'attn_op', op)  # plain attribute


class FlexAttnFuncText(nn.Module):

    text_cross_attention_mask: ClassVar[BlockMask] = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        if FlexAttnFuncText.text_cross_attention_mask is None:
            # Inference fallback (chunk-by-chunk calls supply a single prompt;
            # standard SDPA handles the full-attention case correctly).
            out = F.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
            )
            return out.transpose(1, 2)

        q_varlen = rearrange(query[0], "s n d -> 1 n s d")
        k_varlen = rearrange(key[0], "s n d -> 1 n s d")
        v_varlen = rearrange(value[0], "s n d -> 1 n s d")

        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes

        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        q_varlen = half(q_varlen)
        k_varlen = half(k_varlen)
        v_varlen = half(v_varlen)
        q_varlen = q_varlen.to(v_varlen.dtype)
        k_varlen = k_varlen.to(v_varlen.dtype)

        block_mask = FlexAttnFuncText.text_cross_attention_mask

        x_out = FlexAttnFunc.flex_attn(
            q_varlen,
            k_varlen,
            v_varlen,
            block_mask=block_mask,
            kernel_options={
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_M1": 32,
                "BLOCK_N1": 64,
                "BLOCK_M2": 64,
                "BLOCK_N2": 32,
            },
        )

        x_out = rearrange(x_out, "b n s d -> b s n d")
        return x_out

    @staticmethod
    @torch.no_grad()
    def init_text_mask(
        latent_shape,
        action_shape,
        padded_length,
        chunk_size,
        patch_size,
        text_max_length,
        text_num_chunks,
        device,
    ):
        torch._inductor.config.realize_opcount_threshold = 100
        B, _, L_F, L_H, L_W = latent_shape
        _, _, A_F, A_H, A_W = action_shape

        # Q-side sequence IDs (same ordering as FlexAttnFunc.init_mask:
        # [latent_noisy, latent_clean, action_noisy, action_clean]).
        latent_seq_id = (
            torch.arange(B)[:, None, None, None]
            .expand(-1, L_F // patch_size[0], L_H // patch_size[1], L_W // patch_size[2])
            .flatten()
        )
        action_seq_id = (
            torch.arange(B)[:, None, None, None].expand(-1, A_F, A_H, A_W).flatten()
        )
        seq_ids = torch.cat([latent_seq_id] * 2 + [action_seq_id] * 2)
        seq_ids = F.pad(seq_ids, (0, padded_length), value=-1)

        # Q-side chunk IDs: chunk index of each latent / action token.
        latent_frame_id = (
            torch.arange(L_F)[None, :, None, None]
            .expand(B, -1, L_H // patch_size[1], L_W // patch_size[2])[None]
            .flatten()
        )
        action_frame_id = (
            torch.arange(A_F)[None, :, None, None]
            .expand(B, -1, A_H, A_W)[None]
            .flatten()
        )
        q_chunk_ids = torch.cat(
            [latent_frame_id // chunk_size] * 2 + [action_frame_id // chunk_size] * 2
        )
        q_chunk_ids = F.pad(q_chunk_ids, (0, padded_length), value=-1)

        # KV-side: text tokens laid out as (B, text_num_chunks, text_max_length).
        kv_seq_ids = (
            torch.arange(B)[:, None, None]
            .expand(-1, text_num_chunks, text_max_length)
            .flatten()
        )
        kv_chunk_ids = (
            torch.arange(text_num_chunks)[None, :, None]
            .expand(B, -1, text_max_length)
            .flatten()
        )

        mask_mod = FlexAttnFuncText._get_text_mask_mod(
            seq_ids.long().to(device),
            q_chunk_ids.long().to(device),
            kv_seq_ids.long().to(device),
            kv_chunk_ids.long().to(device),
        )
        block_mask = FlexAttnFunc.compiled_create_block_mask(
            mask_mod,
            1,
            1,
            len(seq_ids),
            len(kv_seq_ids),
            device=device,
            _compile=True,
        )
        FlexAttnFuncText.text_cross_attention_mask = block_mask

    @staticmethod
    @torch.no_grad()
    def _get_text_mask_mod(q_seq_ids, q_chunk_ids, kv_seq_ids, kv_chunk_ids):
        def text_mask(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ):
            same_batch = q_seq_ids[q_idx] == kv_seq_ids[kv_idx]
            same_chunk = q_chunk_ids[q_idx] == kv_chunk_ids[kv_idx]
            valid = (q_seq_ids[q_idx] >= 0) & (kv_seq_ids[kv_idx] >= 0)
            return same_batch & same_chunk & valid

        return text_mask


class FlexAttnFuncEgo(nn.Module):
    """Flex attention op for ego cross-attention with causal temporal masking.

    Each query token at frame *f* can attend to ego KV tokens whose
    ``ego_frame_id < q_frame_id``, implementing causal ego conditioning
    at video-frame resolution.
    """

    ego_cross_attention_mask: ClassVar[BlockMask] = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        if FlexAttnFuncEgo.ego_cross_attention_mask is None:
            # Inference fallback: standard scaled dot-product attention
            out = F.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
            )
            return out.transpose(1, 2)

        q_varlen = rearrange(query[0], "s n d -> 1 n s d")
        k_varlen = rearrange(key[0], "s n d -> 1 n s d")
        v_varlen = rearrange(value[0], "s n d -> 1 n s d")

        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes

        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        q_varlen = half(q_varlen)
        k_varlen = half(k_varlen)
        v_varlen = half(v_varlen)
        q_varlen = q_varlen.to(v_varlen.dtype)
        k_varlen = k_varlen.to(v_varlen.dtype)

        block_mask = FlexAttnFuncEgo.ego_cross_attention_mask

        x_out = FlexAttnFunc.flex_attn(
            q_varlen,
            k_varlen,
            v_varlen,
            block_mask=block_mask,
            kernel_options={
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_M1": 32,
                "BLOCK_N1": 64,
                "BLOCK_M2": 64,
                "BLOCK_N2": 32,
            },
        )

        x_out = rearrange(x_out, "b n s d -> b s n d")
        return x_out

    @staticmethod
    @torch.no_grad()
    def init_ego_mask(
        latent_shape,
        action_shape,
        padded_length,
        chunk_size,
        patch_size,
        ego_num_frames,
        ego_tokens_per_frame,
        device,
    ):
        torch._inductor.config.realize_opcount_threshold = 100
        B, _, L_F, L_H, L_W = latent_shape
        _, _, A_F, A_H, A_W = action_shape

        # -- Q-side: recompute seq_ids / frame_ids (same as FlexAttnFunc.init_mask) --
        latent_seq_id = (
            torch.arange(B)[:, None, None, None]
            .expand(-1, L_F // patch_size[0], L_H // patch_size[1], L_W // patch_size[2])
            .flatten()
        )
        action_seq_id = (
            torch.arange(B)[:, None, None, None].expand(-1, A_F, A_H, A_W).flatten()
        )
        seq_ids = torch.cat([latent_seq_id] * 2 + [action_seq_id] * 2)
        seq_ids = F.pad(seq_ids, (0, padded_length), value=-1)

        latent_frame_id = (
            torch.arange(L_F)[None, :, None, None]
            .expand(B, -1, L_H // patch_size[1], L_W // patch_size[2])[None]
            .flatten()
        )
        action_frame_id = (
            torch.arange(A_F)[None, :, None, None]
            .expand(B, -1, A_H, A_W)[None]
            .flatten()
        )
        frame_ids = torch.cat(
            [latent_frame_id // chunk_size * 2] * 2
            + [action_frame_id // chunk_size * 2 + 1] * 2
        )
        frame_ids = F.pad(frame_ids, (0, padded_length), value=-1)

        # -- KV-side: ego tokens --
        T_video = ego_num_frames
        tpf = ego_tokens_per_frame  # 2

        ego_seq_ids = torch.arange(B)[:, None].expand(-1, T_video * tpf).flatten()

        ego_frame_ids_per_frame = torch.arange(T_video) * 2 + 1  # [1, 3, 5, ..., 2*(N-1)+1]
        ego_frame_per_token = ego_frame_ids_per_frame.repeat_interleave(tpf)  # (T_video * tpf,)
        ego_frame_ids = ego_frame_per_token[None].expand(B, -1).flatten()

        mask_mod = FlexAttnFuncEgo._get_ego_cross_mask_mod(
            seq_ids.long().to(device),
            frame_ids.long().to(device),
            ego_seq_ids.long().to(device),
            ego_frame_ids.long().to(device),
        )
        block_mask = FlexAttnFunc.compiled_create_block_mask(
            mask_mod, 1, 1, len(seq_ids), len(ego_seq_ids), device=device, _compile=True
        )
        FlexAttnFuncEgo.ego_cross_attention_mask = block_mask

    @staticmethod
    @torch.no_grad()
    def _get_ego_cross_mask_mod(q_seq_ids, q_frame_ids, ego_seq_ids, ego_frame_ids):
        def ego_mask(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ):
            same_batch = q_seq_ids[q_idx] == ego_seq_ids[kv_idx]
            valid = (q_seq_ids[q_idx] >= 0) & (ego_seq_ids[kv_idx] >= 0)
            causal = ego_frame_ids[kv_idx] < q_frame_ids[q_idx]
            return same_batch & valid & causal

        return ego_mask


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
            attn_mode="torch",
        )
        # Per-chunk text cross-attention: p_k attends only to chunk-k tokens.
        self.attn2.attn_op = FlexAttnFuncText()
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # Per-frame ego cross-attention using FlexAttention with causal temporal masking.
        self.attn3 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
            attn_mode="torch",
        )
        self.attn3.attn_op = FlexAttnFuncEgo()
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

        # Ego cross-attention: ego_hidden_states is (1, B*T*2, dim) flattened like text
        if ego_hidden_states.shape[1] > 0:
            norm_hidden_states = self.norm4(hidden_states.float()).type_as(hidden_states)
            ego_attn_output = self.attn3(
                norm_hidden_states,
                ego_hidden_states,
                ego_hidden_states,
                None,
                update_cache=0,
                cache_name=cache_name,
            )
            hidden_states = hidden_states + ego_attn_output

        norm_hidden_states = (self.norm3(hidden_states.float()) * (1.0 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states


class WanTransformer3DPhysicalAIModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
        "patch_embedding_mlp",
        "condition_embedder",
        "condition_embedder_action",
        "ego_velocity_acc_mlp",
        "ego_curvature_mlp",
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
        ego_velocity_acc_dim=4,
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
        self.ego_curvature_mlp = nn.Sequential(
            nn.Linear(1, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, inner_dim),
        )

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
        model._init_new_parameters(missing_keys)
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

    def _init_new_parameters(self, missing_keys):
        needs_init = any(
            key.startswith(("ego_velocity_acc_mlp.", "ego_curvature_mlp."))
            or ".attn3." in key
            or ".norm4." in key
            for key in missing_keys
        )

        if not needs_init:
            return

        device = self._get_materialization_device()

        self._materialize_module_if_needed(self.ego_velocity_acc_mlp, device)
        self._materialize_module_if_needed(self.ego_curvature_mlp, device)

        for module in [self.ego_velocity_acc_mlp, self.ego_curvature_mlp]:
            for submodule in module:
                if isinstance(submodule, nn.Linear):
                    nn.init.xavier_uniform_(submodule.weight)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)

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
        curvature = ego_cond["curvature"].to(dtype=self.ego_curvature_mlp[0].weight.dtype)
        if curvature.dim() == 2:
            curvature = curvature.unsqueeze(-1)  # (B, T) -> (B, T, 1)

        B, T = velocity_acc.shape[:2]
        if T == 0:
            inner_dim = self.ego_velocity_acc_mlp[-1].out_features
            return torch.zeros(B, 0, inner_dim, dtype=dtype, device=velocity_acc.device)

        # Per-frame MLP: (B*T, 4) → (B*T, dim) and (B*T, 1) → (B*T, dim)
        vel_embed = self.ego_velocity_acc_mlp(velocity_acc.flatten(0, 1))
        cur_embed = self.ego_curvature_mlp(curvature.flatten(0, 1))
        vel_embed = vel_embed.view(B, T, -1)
        cur_embed = cur_embed.view(B, T, -1)

        # Interleave: [vel_f0, cur_f0, vel_f1, cur_f1, ...]
        ego = torch.stack([vel_embed, cur_embed], dim=2)  # (B, T, 2, dim)
        ego = ego.view(B, T * 2, -1)  # (B, T*2, dim)
        return ego.to(dtype=dtype)

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
        # Per-chunk text embeddings: (B, N_c, L, text_dim) → project → (B, N_c, L, inner_dim).
        text_emb = latent_dict["text_emb"]
        if text_emb.dim() != 4:
            raise ValueError(
                f"Expected per-chunk text_emb of shape (B, N_c, L, D); got {tuple(text_emb.shape)}"
            )
        text_num_chunks = text_emb.shape[1]
        text_max_length = text_emb.shape[2]
        text_hidden_states = self._input_embed(text_emb, input_type="text")
        ego_hidden_states = self._get_ego_hidden_states(latent_dict["ego_cond"], dtype=latent_hidden_states.dtype)

        condition_latent_hidden_states = self._input_embed(latent_dict["latent"], input_type="latent")
        condition_action_hidden_states = self._input_embed(action_dict["latent"], input_type="action")

        latent_hidden_states = latent_hidden_states.flatten(0, 1)[None]
        action_hidden_states = action_hidden_states.flatten(0, 1)[None]
        # (B, N_c, L, inner_dim) → (1, B*N_c*L, inner_dim)
        text_hidden_states = text_hidden_states.flatten(0, 2)[None]
        ego_hidden_states = ego_hidden_states.flatten(0, 1)[None]
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
        ego_num_frames = latent_dict["ego_cond"]["velocity_acc"].shape[1]
        FlexAttnFuncEgo.init_ego_mask(
            latent_shape=latent_dict["noisy_latents"].shape,
            action_shape=action_dict["noisy_latents"].shape,
            padded_length=padded_length,
            chunk_size=input_dict["chunk_size"],
            patch_size=self.patch_size,
            ego_num_frames=ego_num_frames,
            ego_tokens_per_frame=2,
            device=hidden_states.device,
        )
        FlexAttnFuncText.init_text_mask(
            latent_shape=latent_dict["noisy_latents"].shape,
            action_shape=action_dict["noisy_latents"].shape,
            padded_length=padded_length,
            chunk_size=input_dict["chunk_size"],
            patch_size=self.patch_size,
            text_max_length=text_max_length,
            text_num_chunks=text_num_chunks,
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

        # FlexAttn block masks are sized for the training sequence and will
        # mismatch during chunk-by-chunk inference.  Temporarily swap all
        # FlexAttn ops to standard SDPA for the non-train forward path.
        _saved_attn_ops = []
        for block in self.blocks:
            _saved_attn_ops.append((block.attn1.attn_op, block.attn2.attn_op, block.attn3.attn_op))
            for attn in (block.attn1, block.attn2, block.attn3):
                _set_attn_op(attn, _sdpa_for_inference)

        try:
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
        finally:
            for block, (op1, op2, op3) in zip(self.blocks, _saved_attn_ops):
                for attn, op in ((block.attn1, op1), (block.attn2, op2), (block.attn3, op3)):
                    _set_attn_op(attn, op)


# ---------------------------------------------------------------------------
# SelectiveKV KV-cache variants
# ---------------------------------------------------------------------------


class WanTransformerBlockPhysicalAISelectiveKV(nn.Module):
    """PhysicalAI block with SelectiveKV self-attention + per-chunk text + causal ego cross-attn.

    Adds an ``action_mode`` argument that is forwarded to attn1 so the SelectiveKV
    self-attention can choose the right sub-pool (video vs. action).
    """

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
        # 1. Self-attention (SelectiveKV)
        self.attn1 = WanAttentionSelectiveKV(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            attn_mode=attn_mode,
        )

        # 2. Per-chunk text cross-attention: p_k attends only to chunk-k tokens.
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
            attn_mode="torch",
        )
        self.attn2.attn_op = FlexAttnFuncText()
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Per-frame ego cross-attention with causal temporal masking.
        self.attn3 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
            attn_mode="torch",
        )
        self.attn3.attn_op = FlexAttnFuncEgo()
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
        action_mode=False,
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
            action_mode=action_mode,
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

        # Ego cross-attention
        if ego_hidden_states.shape[1] > 0:
            norm_hidden_states = self.norm4(hidden_states.float()).type_as(hidden_states)
            ego_attn_output = self.attn3(
                norm_hidden_states,
                ego_hidden_states,
                ego_hidden_states,
                None,
                update_cache=0,
                cache_name=cache_name,
            )
            hidden_states = hidden_states + ego_attn_output

        norm_hidden_states = (self.norm3(hidden_states.float()) * (1.0 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states


class WanTransformer3DPhysicalAIModelSelectiveKV(ModelMixin, ConfigMixin):
    """PhysicalAI model variant wired up to WanTransformerBlockPhysicalAISelectiveKV.

    Adds SelectiveKV configuration / vis helpers and a two-pool variant of
    ``create_empty_cache`` while keeping ego-conditioning, per-chunk text
    cross-attention, and the inference SDPA fallback intact.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
        "patch_embedding_mlp",
        "condition_embedder",
        "condition_embedder_action",
        "ego_velocity_acc_mlp",
        "ego_curvature_mlp",
        "norm",
    ]
    _no_split_modules = ["WanTransformerBlockPhysicalAISelectiveKV"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm4",
        "norm3",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlockPhysicalAISelectiveKV"]

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
        ego_velocity_acc_dim=4,
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
        self.ego_curvature_mlp = nn.Sequential(
            nn.Linear(1, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, inner_dim),
        )

        self.blocks = nn.ModuleList(
            [
                WanTransformerBlockPhysicalAISelectiveKV(
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
        model._init_new_parameters(missing_keys)
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

    def _init_new_parameters(self, missing_keys):
        needs_init = any(
            key.startswith(("ego_velocity_acc_mlp.", "ego_curvature_mlp."))
            or ".attn3." in key
            or ".norm4." in key
            for key in missing_keys
        )

        if not needs_init:
            return

        device = self._get_materialization_device()

        self._materialize_module_if_needed(self.ego_velocity_acc_mlp, device)
        self._materialize_module_if_needed(self.ego_curvature_mlp, device)

        for module in [self.ego_velocity_acc_mlp, self.ego_curvature_mlp]:
            for submodule in module:
                if isinstance(submodule, nn.Linear):
                    nn.init.xavier_uniform_(submodule.weight)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)

        for block in self.blocks:
            if isinstance(block, WanTransformerBlockPhysicalAISelectiveKV):
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

    def configure_selectiveKV(
        self,
        enabled: bool = True,
        kernel_size: int = 7,
        mix_lambda: float = 0.07,
    ):
        """Enable SelectiveKV eviction (replaces LRU) on all attn1 layers.

            Score(j) = lambda * Relevance(j) - (1 - lambda) * Redundancy(j)

        Lowest-scoring tokens are evicted instead of the oldest ones.
        """
        for i, block in enumerate(self.blocks):
            block.attn1.layer_idx = i
            block.attn1.selectiveKV_enabled = enabled
            block.attn1.selectiveKV_kernel_size = kernel_size
            block.attn1.selectiveKV_lambda = mix_lambda

    def configure_vis(self, layer_indices=None, enabled=True):
        if layer_indices is None:
            layer_indices = list(range(len(self.blocks)))
        for i, block in enumerate(self.blocks):
            block.attn1.layer_idx = i
            if i in layer_indices:
                block.attn1.vis_record = [] if enabled else None
            else:
                block.attn1.vis_record = None

    def reset_vis_records(self):
        for block in self.blocks:
            if block.attn1.vis_record is not None:
                block.attn1.vis_record.clear()

    def collect_vis_records(self):
        records = []
        for block in self.blocks:
            if block.attn1.vis_record:
                records.extend(block.attn1.vis_record)
        return records

    def set_vis_frame_id(self, frame_id: int):
        for block in self.blocks:
            block.attn1._cur_frame_id = frame_id

    def create_empty_cache(self, cache_name, attn_window, latent_token_per_chunk, action_token_per_chunk,
                           device, dtype, batch_size,
                           video_window=None, action_window=None,
                           pin_first_n_video=0, pin_first_n_action=0):
        if video_window is None:
            video_window = attn_window // 2
        if action_window is None:
            action_window = attn_window // 2
        video_tolen = video_window * latent_token_per_chunk
        action_tolen = action_window * action_token_per_chunk
        total_tolen = video_tolen + action_tolen
        for block in self.blocks:
            block.attn1.init_kv_cache(
                cache_name,
                total_tolen,
                self.num_attention_heads,
                self.attention_head_dim,
                device,
                dtype,
                batch_size,
                video_tolen=video_tolen,
                action_tolen=action_tolen,
                pin_first_n_video=pin_first_n_video,
                pin_first_n_action=pin_first_n_action,
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
        curvature = ego_cond["curvature"].to(dtype=self.ego_curvature_mlp[0].weight.dtype)
        if curvature.dim() == 2:
            curvature = curvature.unsqueeze(-1)

        B, T = velocity_acc.shape[:2]
        if T == 0:
            inner_dim = self.ego_velocity_acc_mlp[-1].out_features
            return torch.zeros(B, 0, inner_dim, dtype=dtype, device=velocity_acc.device)

        vel_embed = self.ego_velocity_acc_mlp(velocity_acc.flatten(0, 1))
        cur_embed = self.ego_curvature_mlp(curvature.flatten(0, 1))
        vel_embed = vel_embed.view(B, T, -1)
        cur_embed = cur_embed.view(B, T, -1)

        ego = torch.stack([vel_embed, cur_embed], dim=2)
        ego = ego.view(B, T * 2, -1)
        return ego.to(dtype=dtype)

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
        # Per-chunk text embeddings: (B, N_c, L, text_dim) -> (B, N_c, L, inner_dim).
        text_emb = latent_dict["text_emb"]
        if text_emb.dim() != 4:
            raise ValueError(
                f"Expected per-chunk text_emb of shape (B, N_c, L, D); got {tuple(text_emb.shape)}"
            )
        text_num_chunks = text_emb.shape[1]
        text_max_length = text_emb.shape[2]
        text_hidden_states = self._input_embed(text_emb, input_type="text")
        ego_hidden_states = self._get_ego_hidden_states(latent_dict["ego_cond"], dtype=latent_hidden_states.dtype)

        condition_latent_hidden_states = self._input_embed(latent_dict["latent"], input_type="latent")
        condition_action_hidden_states = self._input_embed(action_dict["latent"], input_type="action")

        latent_hidden_states = latent_hidden_states.flatten(0, 1)[None]
        action_hidden_states = action_hidden_states.flatten(0, 1)[None]
        text_hidden_states = text_hidden_states.flatten(0, 2)[None]
        ego_hidden_states = ego_hidden_states.flatten(0, 1)[None]
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
        ego_num_frames = latent_dict["ego_cond"]["velocity_acc"].shape[1]
        FlexAttnFuncEgo.init_ego_mask(
            latent_shape=latent_dict["noisy_latents"].shape,
            action_shape=action_dict["noisy_latents"].shape,
            padded_length=padded_length,
            chunk_size=input_dict["chunk_size"],
            patch_size=self.patch_size,
            ego_num_frames=ego_num_frames,
            ego_tokens_per_frame=2,
            device=hidden_states.device,
        )
        FlexAttnFuncText.init_text_mask(
            latent_shape=latent_dict["noisy_latents"].shape,
            action_shape=action_dict["noisy_latents"].shape,
            padded_length=padded_length,
            chunk_size=input_dict["chunk_size"],
            patch_size=self.patch_size,
            text_max_length=text_max_length,
            text_num_chunks=text_num_chunks,
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

        # FlexAttn block masks are sized for the training sequence and will
        # mismatch during chunk-by-chunk inference.  Temporarily swap all
        # FlexAttn ops to standard SDPA for the non-train forward path.
        _saved_attn_ops = []
        for block in self.blocks:
            _saved_attn_ops.append((block.attn1.attn_op, block.attn2.attn_op, block.attn3.attn_op))
            for attn in (block.attn1, block.attn2, block.attn3):
                _set_attn_op(attn, _sdpa_for_inference)

        try:
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
                    action_mode=action_mode,
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
        finally:
            for block, (op1, op2, op3) in zip(self.blocks, _saved_attn_ops):
                for attn, op in ((block.attn1, op1), (block.attn2, op2), (block.attn3, op3)):
                    _set_attn_op(attn, op)
