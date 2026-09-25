"""Video-only DiT backbone.

Assumes video-like input at all times:
    1D video: (B, F, C, N)
    2D video: (B, F, C, H, W)

Same interface as :class:`fluidFlow.dit.DiT` (forward, forward_with_cond_scale,
get_2d_params / get_1d_params). ``is_video`` is always True so
``FlowMatching.sample`` can generate noise with a temporal dimension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange, pack, unpack
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Mlp

from .attention import Attention, VisionRotaryEmbeddingFast, LinearAttention, WindowAttention, PhysicsAttention
from .basic_modules import SwiGLUFFN
from .moe import SparseMoeBlock

import math
from functools import partial


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d

def pack_one_with_inverse(x, pattern):
    packed, packed_shape = pack([x], pattern)

    def inverse(x, inverse_pattern=None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]

    return packed, inverse

def project(x, y):
    x, inverse = pack_one_with_inverse(x, 'b *')
    y, _ = pack_one_with_inverse(y, 'b *')
    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = F.normalize(y, dim=-1)
    parallel = (x * unit).sum(dim=-1, keepdim=True) * unit
    orthogonal = x - parallel
    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, bias=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=bias),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=bias),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ConditionEmbedder(nn.Module):
    """
    Embeds scalar conditions into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, cond_dim, hidden_size, dropout_prob):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.null_classes_emb = nn.Parameter(torch.randn(cond_dim))
        self.dropout_prob = dropout_prob

    def token_drop(self, cond_variables, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        batch = cond_variables.shape[0]
        if force_drop_ids is None:
            drop_ids = torch.rand(cond_variables.shape[0], device=cond_variables.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b=batch)
        cond_variables = torch.where(
            rearrange(drop_ids, "b -> b 1"), null_classes_emb, cond_variables
        )
        return cond_variables

    def forward(self, cond_variables, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            cond_variables = self.token_drop(cond_variables, force_drop_ids)
        embeddings = self.mlp(cond_variables)
        return embeddings


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, bias=True, use_swiglu=False, attn_type="vanilla", qk_norm=False, num_experts=None, num_experts_per_tok=None, **attn_kwargs):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=bias)
        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=bias)
        if attn_type == "vanilla":
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=bias, proj_bias=bias, qk_norm=qk_norm, **attn_kwargs)
        elif attn_type == "linear":
            self.attn = LinearAttention(hidden_size, num_heads=num_heads, qkv_bias=bias, proj_bias=bias, qk_norm=qk_norm, **attn_kwargs)
        elif attn_type == "physics":
            self.attn = PhysicsAttention(hidden_size, num_heads=num_heads, qkv_bias=bias, proj_bias=bias, qk_norm=qk_norm, **attn_kwargs)
        else:
            self.attn = None

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if num_experts is not None and num_experts_per_tok is not None:
            self.mlp = SparseMoeBlock(embed_dim=hidden_size, mlp_ratio=mlp_ratio, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        elif use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim), bias=bias)
        else:
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0, bias=bias)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=bias)
        )

    def forward(self, x, c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x.contiguous()
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, N, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, N, C = x.shape
    x = x.view(B, N // window_size, window_size, C)
    windows = x.permute(0, 1, 2, 3).contiguous().view(-1, window_size, C)
    return windows


class WindowBlock(DiTBlock):
    """
    Block for DiT with window attention.
    """
    def __init__(self, hidden_size, num_heads, window_size, shift_size=0, *args, **kwargs):
        super().__init__(hidden_size, num_heads, *args, **kwargs)
        # override attention with window attention
        self.attn = WindowAttention(hidden_size, window_size=window_size, num_heads=num_heads, qkv_bias=False)
        self.window_size = window_size

    def forward(self, x, c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x_windows = window_partition(x, self.window_size)
        attn_windows = self.attn(x_windows).view(x.shape)  # window reverse operation
        x = x + gate_msa.unsqueeze(1) * attn_windows
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, bias=True):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=bias)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=bias)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#                            Video patch embedders                              #
#################################################################################

class PatchEmbed1DVideo(nn.Module):
    """1D video to Patch Embedding: (B, F, C, N) -> (B, F*S, D)."""
    def __init__(self, seq_len, num_frames, patch_size, in_channels, embed_dim):
        super().__init__()
        self.num_frames = num_frames
        self.num_spatial_patches = seq_len // patch_size
        self.num_patches = num_frames * self.num_spatial_patches
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):  # x: (B, F, C, N)
        B, F, C, N = x.shape
        x = x.reshape(B * F, C, N)
        x = self.proj(x).transpose(1, 2)  # (B*F, S, D)
        return x.reshape(B, self.num_patches, -1)  # (B, F*S, D)


class PatchEmbed2DVideo(nn.Module):
    """2D video to Patch Embedding: (B, F, C, H, W) -> (B, F*S, D)."""
    def __init__(self, img_size, num_frames, patch_size, in_channels, embed_dim):
        super().__init__()
        self.num_frames = num_frames
        self._spatial_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.num_spatial_patches = self._spatial_embed.num_patches
        self.num_patches = num_frames * self.num_spatial_patches
        # expose .proj so initialize_weights() can reach it via the same path
        self.proj = self._spatial_embed.proj

    def forward(self, x):  # x: (B, F, C, H, W)
        B, F, C, H, W = x.shape
        x = x.reshape(B * F, C, H, W)
        x = self._spatial_embed(x)  # (B*F, S, D)
        return x.reshape(B, self.num_patches, -1)  # (B, F*S, D)


class FinalLayer1D(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, bias=True):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=bias)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=bias)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SpatialDiTBlock(DiTBlock):
    """
    Standard DiT attention over the spatial dimension only.

    Accepts the full video token sequence (B, F*S, D) but internally folds
    frames into the batch so attention is limited to within-frame patches:
        (B, F*S, D) -> (B*F, S, D) -> attention -> (B, F*S, D)
    """
    def __init__(self, hidden_size, num_heads, *args, num_frames=1, **kwargs):
        super().__init__(hidden_size, num_heads, *args, **kwargs)
        self.num_frames = num_frames

    def forward(self, x, c, feat_rope=None):
        B, FS, D = x.shape
        F, S = self.num_frames, FS // self.num_frames

        # fold frames into batch dimension
        x_s = x.reshape(B * F, S, D)
        c_s = c.unsqueeze(1).expand(-1, F, -1).reshape(B * F, D)

        shift_msa, scale_msa, gate_msa, \
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c_s).chunk(6, dim=1)
        x_s = x_s + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x_s), shift_msa, scale_msa), rope=feat_rope)
        x_s = x_s.contiguous()
        x_s = x_s + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x_s), shift_mlp, scale_mlp))

        return x_s.reshape(B, FS, D)


class TemporalDiTBlock(DiTBlock):
    """
    DiT attention across the temporal (frame) dimension.

    Accepts (B, F*S, D) and transposes to make frames the sequence axis:
        (B, F*S, D) -> (B*S, F, D) -> attention -> (B, F*S, D)
    """
    def __init__(self, hidden_size, num_heads, *args, num_spatial_patches=64, **kwargs):
        super().__init__(hidden_size, num_heads, *args, **kwargs)
        self.num_spatial_patches = num_spatial_patches

    def forward(self, x, c, feat_rope=None):  # feat_rope intentionally ignored
        B, FS, D = x.shape
        S = self.num_spatial_patches
        F = FS // S

        # (B, F, S, D) -> (B*S, F, D)
        x_t = x.reshape(B, F, S, D).permute(0, 2, 1, 3).reshape(B * S, F, D)
        c_t = c.unsqueeze(1).expand(-1, S, -1).reshape(B * S, D)

        shift_msa, scale_msa, gate_msa, \
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c_t).chunk(6, dim=1)
        x_t = x_t + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x_t), shift_msa, scale_msa))  # no RoPE
        x_t = x_t.contiguous()
        x_t = x_t + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x_t), shift_mlp, scale_mlp))

        # (B*S, F, D) -> (B, F*S, D)
        return x_t.reshape(B, S, F, D).permute(0, 2, 1, 3).reshape(B, FS, D)


#################################################################################
#                                 Video-only DiT                                #
#################################################################################

class VideoDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone, video-only.

    Supported input shapes
    ----------------------
    Video 1-D  (B, F, C, N)
    Video 2-D  (B, F, C, H, W)

    Same interface as :class:`fluidFlow.dit.DiT`.
    """
    def __init__(
        self,
        input_size=1024,
        num_frames=8,
        patch_size=16,
        in_channels=1,
        cond_dim=2,
        class_dropout_prob=0.1,
        hidden_size=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        learn_sigma=False,
        use_bias=True,
        use_swiglu=False,
        use_rope=False,
        attn_type="vanilla",
        slice_num=128,
        window_size=64,
        qk_norm=False,
        num_experts=None,
        num_experts_per_tok=None,
        factorize=True,
        **kwargs
    ):
        super().__init__()
        assert num_frames is not None and num_frames > 1, "VideoDiT requires num_frames > 1"
        self.learn_sigma = learn_sigma
        self.channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim
        self.self_condition = False  # Not used in DiT, here for interface compatibility
        self.num_frames = num_frames
        self.is_video = True
        self.factorize = factorize

        # -- Patch embedder & final layer --
        self.input_size = input_size
        if isinstance(input_size, int):
            self.is_1d = True
            self.x_embedder = PatchEmbed1DVideo(
                input_size, num_frames, patch_size, in_channels, hidden_size)
            self.final_layer = FinalLayer1D(hidden_size, patch_size,
                                            self.out_channels, bias=use_bias)
            print("Creating Video 1D DiT")
        else:
            assert isinstance(input_size, tuple) and len(input_size) == 2
            self.is_1d = False
            self.x_embedder = PatchEmbed2DVideo(
                input_size, num_frames, patch_size, in_channels, hidden_size)
            self.final_layer = FinalLayer(hidden_size, patch_size,
                                          self.out_channels, bias=use_bias)
            print("Creating Video 2D DiT")

        # -- RoPE (spatial only) --
        if use_rope and self.is_1d:
            head_dim = hidden_size // num_heads
            seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(dim=head_dim, max_seq_len=seq_len)
        else:
            if use_rope and not self.is_1d:
                print("rope is only implemented for 1D DiT. Setting use_rope to False.")
            self.feat_rope = None

        # -- Condition / timestep embedders --
        self.t_embedder = TimestepEmbedder(hidden_size, bias=use_bias)
        self.y_embedder = ConditionEmbedder(cond_dim, hidden_size, class_dropout_prob)

        num_spatial_patches = self.x_embedder.num_spatial_patches
        num_patches = self.x_embedder.num_patches
        print(f"Creating VideoDiT with {num_patches} total patches "
              f"({'factorized' if factorize else 'full-attn'} mode).")

        # -- Transformer blocks --
        if factorize:
            # Alternate: even indices -> SpatialDiTBlock, odd -> TemporalDiTBlock
            self.blocks = nn.ModuleList()
            for i in range(depth):
                if i % 2 == 0:
                    blk = SpatialDiTBlock(
                        hidden_size, num_heads,
                        num_frames=num_frames,
                        mlp_ratio=mlp_ratio, bias=use_bias,
                        use_swiglu=use_swiglu, attn_type=attn_type,
                        qk_norm=qk_norm, num_experts=num_experts,
                        num_experts_per_tok=num_experts_per_tok,
                        slice_num=slice_num)
                else:
                    blk = TemporalDiTBlock(
                        hidden_size, num_heads,
                        num_spatial_patches=num_spatial_patches,
                        mlp_ratio=mlp_ratio, bias=use_bias,
                        use_swiglu=use_swiglu,
                        attn_type="vanilla",  # temporal always uses vanilla attn
                        qk_norm=qk_norm)
                self.blocks.append(blk)
        else:
            # Full attention over all F*S tokens
            block_class = (partial(WindowBlock, window_size=window_size)
                           if attn_type == "window" else DiTBlock)
            self.blocks = nn.ModuleList([
                block_class(
                    hidden_size, num_heads,
                    mlp_ratio=mlp_ratio, bias=use_bias,
                    use_swiglu=use_swiglu, attn_type=attn_type,
                    qk_norm=qk_norm, num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    slice_num=slice_num)
                for _ in range(depth)
            ])

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Spatial positional embedding: (1, S, D), S = patches-per-frame
        pos_embed = get_1d_sincos_pos_embed(
            self.hidden_size,
            self.x_embedder.num_spatial_patches
        )
        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0))  # (1, S, D)

        # Temporal positional embedding: (1, F, D)
        temp_embed = get_1d_sincos_pos_embed(self.hidden_size, self.num_frames)
        self.register_buffer("temporal_pos_embed",
                             torch.from_numpy(temp_embed).float().unsqueeze(0))  # (1, F, D)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.mlp[2].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _video_pos_embed(self):
        """
        Build (1, F*S, D) by adding broadcast spatial and temporal embeddings.

        Token layout: [f0_s0 .. f0_sS | f1_s0 .. f1_sS | .. | f(F-1)_s0 .. f(F-1)_sS]
        """
        S = self.x_embedder.num_spatial_patches
        F = self.num_frames
        D = self.hidden_size

        # (1, S, D) -> (1, F, S, D) -> (1, F*S, D)
        sp = self.pos_embed.unsqueeze(1).expand(-1, F, -1, -1).reshape(1, F * S, D)
        # (1, F, D) -> (1, F, S, D) -> (1, F*S, D)
        tp = self.temporal_pos_embed.unsqueeze(2).expand(-1, -1, S, -1).reshape(1, F * S, D)
        return sp + tp

    def unpatchify(self, x):
        """
        x: (B, F*S, p*C) for 1D or (B, F*S, p^2*C) for 2D
        output: (B, F, C, S) for 1D or (B, F, C, H, W) for 2D
        """
        c = self.out_channels
        p = self.patch_size
        F = self.num_frames
        S = x.shape[1] // F  # spatial patches per frame

        if self.is_1d:
            x = x.reshape(x.shape[0], F, S, p, c)
            x = x.permute(0, 1, 4, 2, 3)  # (B, F, C, S, p)
            x = x.reshape(x.shape[0], F, c, S * p)  # (B, F, C, seq_len)
        else:
            h = self.input_size[0] // p
            w = self.input_size[1] // p
            x = x.reshape(x.shape[0], F, h, w, p, p, c)
            x = x.permute(0, 1, 6, 2, 4, 3, 5)  # (B, F, C, h, p, w, p)
            x = x.reshape(x.shape[0], F, c, h * p, w * p)  # (B, F, C, H, W)

        return x

    def forward(self, x, t, classes, return_act=False, *args, **kwargs):
        """
        Forward pass of VideoDiT.
        x: (B, F, C, N) for 1D video or (B, F, C, H, W) for 2D video
        t: (B,) tensor of diffusion timesteps
        classes: (B, cond_dim) tensor of conditioning variables
        """
        x = self.x_embedder(x) + self._video_pos_embed()  # (B, F*S, D)
        t = self.t_embedder(t)  # (N, D)
        force_drop_ids = kwargs.get("force_drop_ids", None)
        y = self.y_embedder(classes, self.training, force_drop_ids)  # (N, D)
        c = t + y  # (N, D)
        for block in self.blocks:
            x = block(x, c, self.feat_rope)  # (N, T, D)
        act = x
        x = self.final_layer(x, c)  # (B, F*S, patch_size * out_channels)
        x = self.unpatchify(x)  # (B, F, out_channels, ...)
        if return_act:
            return x, act
        return x

    def forward_with_cond_scale(
        self,
        x,
        t,
        classes,
        cond_scale=6,
        rescaled_phi=0.7,
        remove_parallel_component=True,
        keep_parallel_frac=0,
        cfg_interval_start=0,
        *args,
        **kwargs,
    ):
        """
        Forward pass of VideoDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        batch_size = x.shape[0]
        combined = torch.cat([x, x], dim=0)
        force_drop_ids = torch.cat(
            [
                torch.zeros((batch_size,), dtype=torch.bool, device=x.device),
                torch.ones((batch_size,), dtype=torch.bool, device=x.device),
            ],
            dim=0,
        )
        y_combined = torch.cat([classes, classes], dim=0)
        t_combined = torch.cat([t, t], dim=0)
        model_out = self.forward(combined, t_combined, y_combined, force_drop_ids=force_drop_ids)
        # Video layout is (B, F, C, ...) so the channel dim is 2
        ch = self.channels
        cdim = 2
        eps = model_out.narrow(cdim, 0, ch)
        rest = model_out.narrow(cdim, ch, self.out_channels - ch)
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        update = cond_eps - uncond_eps
        if remove_parallel_component:
            parallel, orthog = project(update, cond_eps)
            update = orthog + parallel * keep_parallel_frac
        half_eps = cond_eps + update * (cond_scale - 1)

        if cfg_interval_start > 0:
            timestep = t[0]
            if timestep < cfg_interval_start:
                half_eps = cond_eps

        if rescaled_phi != 0:
            std_fn = partial(torch.std, dim=tuple(range(1, half_eps.ndim)), keepdim=True)
            rescaled_logits = half_eps * (std_fn(cond_eps) / std_fn(half_eps))
            half_eps = rescaled_logits * rescaled_phi + half_eps * (1. - rescaled_phi)

        eps = torch.cat([half_eps, uncond_eps], dim=0)
        eps_sigma = torch.cat([eps, rest], dim=cdim)
        # return cfg eps
        return eps_sigma.chunk(2, dim=0)[0]

    def get_2d_params(self):
        """
        Return parameters suitable for Muon optimizer (2D+ parameters like weight matrices).
        """
        return [p for p in self.parameters() if p.dim() == 2]

    def get_1d_params(self):
        """
        Return parameters not suitable for Muon optimizer (1D parameters like biases).
        """
        return [p for p in self.parameters() if p.dim() != 2]


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width, or tuple (height, width)
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
    else:
        grid_h = np.arange(grid_size[0], dtype=np.float32)
        grid_w = np.arange(grid_size[1], dtype=np.float32)

    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_h.shape[0], grid_w.shape[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    """
    length: int, number of positions
    return: (length, embed_dim)
    """
    pos = np.arange(length, dtype=np.float32)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


#################################################################################
#                               VideoDiT Configs                                #
#################################################################################

def VideoDiT_XL_1(**kwargs):
    return VideoDiT(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)

def VideoDiT_XL_2(**kwargs):
    return VideoDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def VideoDiT_XL_4(**kwargs):
    return VideoDiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def VideoDiT_XL_8(**kwargs):
    return VideoDiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def VideoDiT_L_1(**kwargs):
    return VideoDiT(depth=24, hidden_size=1024, patch_size=1, num_heads=16, **kwargs)

def VideoDiT_L_2(**kwargs):
    return VideoDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def VideoDiT_L_4(**kwargs):
    return VideoDiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def VideoDiT_L_8(**kwargs):
    return VideoDiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def VideoDiT_B_1(**kwargs):
    return VideoDiT(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def VideoDiT_B_2(**kwargs):
    return VideoDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def VideoDiT_B_4(**kwargs):
    return VideoDiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def VideoDiT_B_8(**kwargs):
    return VideoDiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def VideoDiT_S_1(**kwargs):
    return VideoDiT(depth=12, hidden_size=384, patch_size=1, num_heads=6, **kwargs)

def VideoDiT_S_2(**kwargs):
    return VideoDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def VideoDiT_S_4(**kwargs):
    return VideoDiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def VideoDiT_S_8(**kwargs):
    return VideoDiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def VideoDiT_XS_1(**kwargs):
    return VideoDiT(depth=8, hidden_size=256, patch_size=1, num_heads=4, **kwargs)

def VideoDiT_XS_2(**kwargs):
    return VideoDiT(depth=8, hidden_size=256, patch_size=2, num_heads=4, **kwargs)

def VideoDiT_XS_4(**kwargs):
    return VideoDiT(depth=8, hidden_size=256, patch_size=4, num_heads=4, **kwargs)

def VideoDiT_XS_8(**kwargs):
    return VideoDiT(depth=8, hidden_size=256, patch_size=8, num_heads=4, **kwargs)

def VideoDiT_XXS_1(**kwargs):
    return VideoDiT(depth=6, hidden_size=128, patch_size=1, num_heads=4, **kwargs)

def VideoDiT_XXS_2(**kwargs):
    return VideoDiT(depth=6, hidden_size=128, patch_size=2, num_heads=4, **kwargs)

def VideoDiT_XXS_4(**kwargs):
    return VideoDiT(depth=6, hidden_size=128, patch_size=4, num_heads=4, **kwargs)

def VideoDiT_XXS_8(**kwargs):
    return VideoDiT(depth=6, hidden_size=128, patch_size=8, num_heads=4, **kwargs)

def VideoDiT_XXXS_1(**kwargs):
    return VideoDiT(depth=4, hidden_size=128, patch_size=1, num_heads=4, **kwargs)

VideoDiT_models = {
    'VideoDiT-XL/2': VideoDiT_XL_2, 'VideoDiT-XL/1': VideoDiT_XL_1, 'VideoDiT-XL/4': VideoDiT_XL_4, 'VideoDiT-XL/8': VideoDiT_XL_8,
    'VideoDiT-L/2': VideoDiT_L_2, 'VideoDiT-L/1': VideoDiT_L_1, 'VideoDiT-L/4': VideoDiT_L_4, 'VideoDiT-L/8': VideoDiT_L_8,
    'VideoDiT-B/2': VideoDiT_B_2, 'VideoDiT-B/1': VideoDiT_B_1, 'VideoDiT-B/4': VideoDiT_B_4, 'VideoDiT-B/8': VideoDiT_B_8,
    'VideoDiT-S/2': VideoDiT_S_2, 'VideoDiT-S/1': VideoDiT_S_1, 'VideoDiT-S/4': VideoDiT_S_4, 'VideoDiT-S/8': VideoDiT_S_8,
    'VideoDiT-XS/1': VideoDiT_XS_1, 'VideoDiT-XS/2': VideoDiT_XS_2, 'VideoDiT-XS/4': VideoDiT_XS_4, 'VideoDiT-XS/8': VideoDiT_XS_8,
    'VideoDiT-XXS/1': VideoDiT_XXS_1, 'VideoDiT-XXS/2': VideoDiT_XXS_2, 'VideoDiT-XXS/4': VideoDiT_XXS_4, 'VideoDiT-XXS/8': VideoDiT_XXS_8,
    'VideoDiT-XXXS/1': VideoDiT_XXXS_1
}
