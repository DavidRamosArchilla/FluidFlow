import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import repeat, rearrange, pack, unpack
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Mlp

from .attention import Attention, VisionRotaryEmbeddingFast, LinearAttention, WindowAttention
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

    def inverse(x, inverse_pattern = None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]

    return packed, inverse

def project(x, y):
    x, inverse = pack_one_with_inverse(x, 'b *')
    y, _ = pack_one_with_inverse(y, 'b *')

    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = F.normalize(y, dim = -1)

    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
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
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
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
        # self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        # self.cond_dim = cond_dim
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
        null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)
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
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=bias) # nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # attention_class = LinearAttention if linear_attn else Attention
        # self.attn = attention_class(hidden_size, num_heads=num_heads, qkv_bias=bias, proj_bias=bias, qk_norm=qk_norm, **attn_kwargs)
        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=bias) # nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if attn_type == "vanilla":
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=bias, proj_bias=bias, qk_norm=qk_norm, **attn_kwargs)
        elif attn_type == "linear":
            self.attn = LinearAttention(hidden_size, num_heads=num_heads, qkv_bias=bias, proj_bias=bias, qk_norm=qk_norm, **attn_kwargs)
        else:
            self.attn = None
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if num_experts is not None and num_experts_per_tok is not None:
            self.mlp = SparseMoeBlock(embed_dim=hidden_size, mlp_ratio=mlp_ratio, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok) # SparseMoeBlock
        elif use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim), bias=bias)
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
    # x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
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
        # shift_size will be ignored for the moment

    def forward(self, x, c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x_windows = window_partition(x, self.window_size) 
        attn_windows = self.attn(x_windows).view(x.shape) # window reverse operation
        x = x + gate_msa.unsqueeze(1) * attn_windows
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, bias=True):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=bias) # nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
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
#                                     1D DiT                                    #
#################################################################################

class PatchEmbed1D(nn.Module):
    """1D sequence to Patch Embedding"""
    def __init__(self, seq_len, patch_size, in_channels, embed_dim):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.proj = nn.Conv1d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        # x: (B, C, S)
        x = self.proj(x)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x
    
class FinalLayer1D(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, bias=True):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=bias) # nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
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
    

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=1024,        
        patch_size=16,       # Patch size along sequence
        in_channels=1,
        cond_dim=2,
        class_dropout_prob=0.1,
        hidden_size=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        learn_sigma=False,
        use_bias=True, # this is to use muon
        use_swiglu=False,
        use_rope=False,
        attn_type="vanilla",
        window_size=64,
        qk_norm=False,
        num_experts=None,
        num_experts_per_tok=None,
        **kwargs
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.cond_dim = cond_dim
        self.self_condition = False  # Not used in DiT, this is here for interface compatibility
        if isinstance(input_size, int):
            self.x_embedder = PatchEmbed1D(input_size, patch_size, in_channels, hidden_size)
            self.final_layer = FinalLayer1D(hidden_size, patch_size, self.out_channels, bias=use_bias)
            self.is_1d = True
            print("Creating 1D DiT")
        else:
            assert isinstance(input_size, tuple) and len(input_size) == 2
            self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
            self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, bias=use_bias)
            self.is_1d = False
            print("Creating 2D DiT")

        if use_rope:
            if self.is_1d:
                head_dim = hidden_size // num_heads
                seq_len = input_size // patch_size
                self.feat_rope = VisionRotaryEmbeddingFast(
                    dim=head_dim,
                    max_seq_len=seq_len,
                )
            else:
                print("rope is only implemente for 1D DiT for the moment. Setting use_rope to False.")
                self.feat_rope = None
        else:
            self.feat_rope = None

        self.t_embedder = TimestepEmbedder(hidden_size, bias=use_bias)
        self.y_embedder = ConditionEmbedder(cond_dim, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        print(f"Creating DiT with {num_patches} patches.")
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        block_class = partial(WindowBlock, window_size=window_size) if attn_type == "window" else DiTBlock
        self.blocks = nn.ModuleList(
            [
                block_class(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    bias=use_bias,
                    use_swiglu=use_swiglu,
                    attn_type=attn_type,
                    qk_norm=qk_norm,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                )
                for _ in range(depth)
            ]
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            self.x_embedder.num_patches
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

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

    
    def unpatchify(self, x):
        """
        x: (B, num_patches, patch_size * C) for 1D or (B, num_patches, patch_size^2 * C) for 2D
        output: (B, C, S) for 1D or (B, C, H, W) for 2D
        """
        c = self.out_channels
        p = self.patch_size
        num_patches = x.shape[1]
        
        if self.is_1d:
            # 1D case: output (B, C, S)
            x = x.reshape(x.shape[0], num_patches, p, c)  # (B, num_patches, patch_size, C)
            x = x.permute(0, 3, 1, 2)  # (B, C, num_patches, patch_size)
            x = x.reshape(x.shape[0], c, num_patches * p)  # (B, C, S)
        else:
            # 2D case: output (B, C, H, W)
            h = w = int(x.shape[1] ** 0.5)
            assert h * w == x.shape[1]
            x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
            x = torch.einsum('nhwpqc->nchpwq', x)
            x = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        
        return x

    def forward(self, x, t, classes, return_act=False, *args, **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        force_drop_ids = kwargs.get("force_drop_ids", None)
        y = self.y_embedder(classes, self.training, force_drop_ids)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            # x = checkpoint(block, x, c, self.feat_rope, use_reentrant=False)
            x = block(x, c, self.feat_rope)                      # (N, T, D)
        act = x
        x = self.final_layer(x, c)               # (B, num_patches, patch_size * out_channels)
        x = self.unpatchify(x)                   # (B, out_channels, S)
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
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        # half = x[: len(x) // 2]
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
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # separate noise predictions and from variance predictions if present
        eps, rest = model_out[:, :self.channels], model_out[:, self.channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        # return half_eps, uncond_eps
        update = cond_eps - uncond_eps
        if remove_parallel_component:
            parallel, orthog = project(update, cond_eps)
            update = orthog + parallel * keep_parallel_frac
        # half_eps = uncond_eps + cond_scale * (cond_eps - uncond_eps)
        half_eps = cond_eps + update * (cond_scale - 1)

        if cfg_interval_start > 0:
            timestep = t[0]
            if timestep < cfg_interval_start:
                half_eps = cond_eps

        if rescaled_phi != 0:
            std_fn = partial(torch.std, dim = tuple(range(1, half_eps.ndim)), keepdim = True)
            rescaled_logits = half_eps * (std_fn(cond_eps) / std_fn(half_eps))
            half_eps = rescaled_logits * rescaled_phi + half_eps * (1. - rescaled_phi)

        eps = torch.cat([half_eps, uncond_eps], dim=0)
        eps_sigma = torch.cat([eps, rest], dim=1)
        # # return cfg eps and unconditioned eps
        return eps_sigma.chunk(2, dim=0)[0]


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

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
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

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

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
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_1(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_1(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=1, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_1(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_1(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=1, num_heads=6, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def DiT_XS_1(**kwargs):
    return DiT(depth=8, hidden_size=256, patch_size=1, num_heads=4, **kwargs)

def DiT_XS_2(**kwargs):
    return DiT(depth=8, hidden_size=256, patch_size=2, num_heads=4, **kwargs)

def DiT_XS_4(**kwargs):
    return DiT(depth=8, hidden_size=256, patch_size=4, num_heads=4, **kwargs)

def DiT_XS_8(**kwargs):
    return DiT(depth=8, hidden_size=256, patch_size=8, num_heads=4, **kwargs)

def DiT_XXS_1(**kwargs):
    return DiT(depth=6, hidden_size=128, patch_size=1, num_heads=4, **kwargs)

def DiT_XXS_2(**kwargs):
    return DiT(depth=6, hidden_size=128, patch_size=2, num_heads=4, **kwargs)

def DiT_XXS_4(**kwargs):
    return DiT(depth=6, hidden_size=128, patch_size=4, num_heads=4, **kwargs)

def DiT_XXS_8(**kwargs):
    return DiT(depth=6, hidden_size=128, patch_size=8, num_heads=4, **kwargs)

def DiT_XXXS_1(**kwargs):
    return DiT(depth=4, hidden_size=128, patch_size=1, num_heads=4, **kwargs)

DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/1': DiT_XL_1,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/1':  DiT_L_1,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/1':  DiT_B_1,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/1':  DiT_S_1,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
    'DiT-XS/1': DiT_XS_1,  'DiT-XS/2': DiT_XS_2,  'DiT-XS/4': DiT_XS_4,  'DiT-XS/8': DiT_XS_8,
    'DiT-XXS/1': DiT_XXS_1, 'DiT-XXS/2': DiT_XXS_2, 'DiT-XXS/4': DiT_XXS_4, 'DiT-XXS/8': DiT_XXS_8,
    'DiT-XXXS/1': DiT_XXXS_1
}