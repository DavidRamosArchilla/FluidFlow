"""LTX-2 style video VAE adapted to 1D-mesh unsteady CFD data.

Reference: https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-core/src/ltx_core/model/video_vae/video_vae.py
(plus ``conv_video_decoder.py``, ``resnet.py``, ``sampling.py``, ``ops.py``,
``attention.py``, ``convolution.py`` from the same package).

What is adapted here
--------------------
LTX works on videos ``(B, C, F, H, W)`` with 3D convolutions over
``(time, height, width)``.  Our data are unsteady CFD simulations shaped::

    (N_simulations, N_frames, channels, mesh_points)

i.e. each frame of each simulation is a **1D sequence** over the mesh.
This module therefore implements the same architecture with 2D convolutions
over ``(time, mesh)`` on tensors shaped ``(B, C, F, L)``:

* ``patchify`` compresses the mesh axis only: ``L -> L / patch``.
* ``compress_time``       : stride ``(2, 1)`` over ``(F, L)`` (causal in time).
* ``compress_mesh``       : stride ``(1, 2)`` over ``(F, L)``.
* ``compress_all``        : stride ``(2, 2)`` over ``(F, L)``.
* ``*_res`` variants use the LTX ``SpaceToDepth`` residual downsampler.
* Temporal downsampling is causal (repeat-first-frame padding), so with a
  total time factor of 8 the latent has ``F' = 1 + (F - 1) / 8`` frames and
  the input must satisfy ``F = 1 + 8 * k`` (extra frames are cropped with a
  warning, exactly like LTX).
* The mesh axis must be divisible by the total mesh factor (32 by default);
  pad it beforehand (see ``scripts/train_vae_ltx_airfoil.py``).

What is deliberately ignored from LTX-2
---------------------------------------
Anything multimodal (text/audio conditioning), tiling, distributed decoding,
keyframe-anchored diffusion decoding and GAN/perceptual losses.  This is a
plain convolutional KL-VAE with an MSE + KL loss, API-compatible with
``TrainerVAE1D`` in ``denoising_diffusion_pytorch.autoencoder`` and with the
latent-saving utilities in ``scripts/train_vae_cylinder.py``.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _safe_groups(channels: int, want: int = 32) -> int:
    """Largest divisor of ``channels`` that is <= ``want`` (GroupNorm safe)."""
    for g in range(min(want, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class PixelNorm(nn.Module):
    """Per-location RMS norm over channels (LTX default norm layer)."""

    def __init__(self, dim: int = 1, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=self.dim, keepdim=True) + self.eps)
        return x / rms


def _make_norm(channels: int, norm_layer: str, num_groups: int = 32) -> nn.Module:
    if norm_layer == "group_norm":
        return nn.GroupNorm(_safe_groups(channels, num_groups), channels, eps=1e-6, affine=True)
    if norm_layer == "pixel_norm":
        return PixelNorm()
    raise ValueError(f"unknown norm_layer {norm_layer!r} (use 'group_norm' or 'pixel_norm')")


# ---------------------------------------------------------------------------
# causal convolution over (time, mesh)
# ---------------------------------------------------------------------------

class CausalConv2d(nn.Module):
    """2D conv over ``(F, L)`` with causal padding in time.

    Causal mode repeats the first frame ``k_t - 1`` times (like LTX
    ``CausalConv3d``), so frame ``t`` of the output only sees frames
    ``<= t`` of the input.  Non-causal mode pads symmetrically on both sides
    and is used by the decoder (LTX default ``causal=False`` there).
    The mesh axis is padded symmetrically in both modes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: Tuple[int, int] = (1, 1),
        groups: int = 1,
        bias: bool = True,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.time_kernel = kernel_size[0]
        pad_l = kernel_size[1] // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(0, pad_l),
            padding_mode=spatial_padding_mode,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        k = self.time_kernel
        if k > 1:
            if causal:
                pad = x[:, :, :1, :].repeat(1, 1, k - 1, 1)
                x = torch.cat([pad, x], dim=2)
            else:
                pre = x[:, :, :1, :].repeat(1, 1, (k - 1) // 2, 1)
                post = x[:, :, -1:, :].repeat(1, 1, (k - 1) // 2, 1)
                x = torch.cat([pre, x, post], dim=2)
        return self.conv(x)

    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight


# ---------------------------------------------------------------------------
# resnet / mid / attention blocks (LTX resnet.py + attention.py, 2D version)
# ---------------------------------------------------------------------------

class ResnetBlock2D(nn.Module):
    """LTX ``ResnetBlock3D`` with 2D causal convs over ``(F, L)``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-6,
        norm_layer: str = "pixel_norm",
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = _make_norm(in_channels, norm_layer, groups)
        self.act = nn.SiLU()
        self.conv1 = CausalConv2d(in_channels, out_channels, 3,
                                  spatial_padding_mode=spatial_padding_mode)
        self.norm2 = _make_norm(out_channels, norm_layer, groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CausalConv2d(out_channels, out_channels, 3,
                                  spatial_padding_mode=spatial_padding_mode)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
        # GroupNorm(1 group) ~= LayerNorm, avoids rearrange (same trick as LTX)
        self.norm_shortcut = nn.GroupNorm(1, in_channels, eps=eps, affine=True) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h, causal=causal)
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h, causal=causal)
        return self.shortcut(self.norm_shortcut(x)) + h


class UNetMidBlock2D(nn.Module):
    """Stack of ``num_layers`` ResnetBlock2D (LTX ``UNetMidBlock3D``)."""

    def __init__(self, in_channels: int, num_layers: int = 1, dropout: float = 0.0,
                 resnet_groups: int = 32, norm_layer: str = "pixel_norm",
                 spatial_padding_mode: str = "zeros"):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResnetBlock2D(in_channels, in_channels, dropout=dropout,
                          groups=resnet_groups, norm_layer=norm_layer,
                          spatial_padding_mode=spatial_padding_mode)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, causal=causal)
        return x


class AttnBlock2D(nn.Module):
    """Per-frame self-attention over the mesh axis.

    LTX ``AttnBlock3D`` folds frames into the batch and attends over ``H*W``
    per frame with no cross-frame interaction.  Here frames are folded into
    the batch and attention runs over ``L`` (mesh points) per frame.
    """

    def __init__(self, channels: int, num_heads: int = 1, qk_norm: bool = True):
        super().__init__()
        assert channels % num_heads == 0, f"{channels=} not divisible by {num_heads=}"
        self.num_heads = num_heads
        self.dim_head = channels // num_heads
        self.norm = PixelNorm()
        self.to_qkv = nn.Conv1d(channels, channels * 3, 1, bias=False)
        self.q_norm = nn.RMSNorm(self.dim_head) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.dim_head) if qk_norm else nn.Identity()
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:  # noqa: ARG002
        identity = x
        b, c, f, length = x.shape
        h = rearrange(x, "b c f l -> (b f) c l")
        h = self.norm(h)
        q, k, v = self.to_qkv(h).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "bf (hd d) l -> bf hd l d", hd=self.num_heads), (q, k, v))
        dtype = q.dtype
        q = self.q_norm(q.to(self.q_norm.weight.dtype)).to(dtype) if not isinstance(self.q_norm, nn.Identity) else q
        k = self.k_norm(k.to(self.k_norm.weight.dtype)).to(dtype) if not isinstance(self.k_norm, nn.Identity) else k
        h = F.scaled_dot_product_attention(q, k, v)
        h = rearrange(h, "bf hd l d -> bf (hd d) l")
        h = self.proj(h)
        return identity + rearrange(h, "(b f) c l -> b c f l", b=b)


# ---------------------------------------------------------------------------
# down / up sampling (LTX sampling.py, 2D version)
# ---------------------------------------------------------------------------

class SpaceToDepthDownsample2D(nn.Module):
    """LTX ``SpaceToDepthDownsample`` over ``(F, L)``.

    Residual downsampler: a strided space-to-depth skip (mean over the group)
    plus a causal conv branch rearranged the same way.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: Tuple[int, int],
                 spatial_padding_mode: str = "zeros"):
        super().__init__()
        self.stride = stride
        self.group_size = in_channels * math.prod(stride) // out_channels
        assert in_channels * math.prod(stride) % out_channels == 0
        self.conv = CausalConv2d(in_channels, out_channels // math.prod(stride), 3,
                                 spatial_padding_mode=spatial_padding_mode)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        st, sl = self.stride
        if st == 2:
            x = torch.cat([x[:, :, :1, :], x], dim=2)  # duplicate first frame
        skip = rearrange(x, "b c (f p1) (l p2) -> b (c p1 p2) f l", p1=st, p2=sl)
        skip = rearrange(skip, "b (c g) f l -> b c g f l", g=self.group_size).mean(dim=2)
        h = self.conv(x, causal=causal)
        h = rearrange(h, "b c (f p1) (l p2) -> b (c p1 p2) f l", p1=st, p2=sl)
        return h + skip


class DepthToSpaceUpsample2D(nn.Module):
    """LTX ``DepthToSpaceUpsample`` over ``(F, L)`` (decoder mirror)."""

    def __init__(self, in_channels: int, stride: Tuple[int, int],
                 out_channels_reduction_factor: int = 1, residual: bool = False,
                 spatial_padding_mode: str = "reflect"):
        super().__init__()
        self.stride = stride
        self.residual = residual
        self.reduction = out_channels_reduction_factor
        out_ch = math.prod(stride) * in_channels // out_channels_reduction_factor
        self.conv = CausalConv2d(in_channels, out_ch, 3,
                                 spatial_padding_mode=spatial_padding_mode)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        st, sl = self.stride
        if self.residual:
            skip = rearrange(x, "b (c p1 p2) f l -> b c (f p1) (l p2)", p1=st, p2=sl)
            skip = skip.repeat(1, math.prod(self.stride) // self.reduction, 1, 1)
            if st == 2:
                skip = skip[:, :, 1:, :]
        h = self.conv(x, causal=causal)
        h = rearrange(h, "b (c p1 p2) f l -> b c (f p1) (l p2)", p1=st, p2=sl)
        if st == 2:
            h = h[:, :, 1:, :]  # drop the duplicated first frame
        return h + skip if self.residual else h


# ---------------------------------------------------------------------------
# patchify / per-channel statistics (LTX ops.py, mesh-axis version)
# ---------------------------------------------------------------------------

def patchify_mesh(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """``(B, C, F, L) -> (B, C * patch, F, L / patch)`` (LTX patchify, mesh only)."""
    if patch_size == 1:
        return x
    return rearrange(x, "b c f (l p) -> b (c p) f l", p=patch_size)


def unpatchify_mesh(x: torch.Tensor, patch_size: int, channels: int) -> torch.Tensor:
    """Inverse of :func:`patchify_mesh`."""
    if patch_size == 1:
        return x
    return rearrange(x, "b (c p) f l -> b c f (l p)", p=patch_size, c=channels)


class PerChannelStatistics(nn.Module):
    """Dataset statistics for normalising/denormalising latents (LTX ``ops.py``).

    Buffers are identity by default so a model without a fitted checkpoint is
    a no-op.  The training script in ``scripts/train_vae_ltx_airfoil.py``
    follows ``train_vae_cylinder.py`` instead (external standardisation saved
    to ``latents_stats.npz``); use :meth:`set_statistics` only if you want the
    LTX-style built-in normalisation.
    """

    def __init__(self, latent_channels: int):
        super().__init__()
        self.register_buffer("std-of-means", torch.ones(latent_channels))
        self.register_buffer("mean-of-means", torch.zeros(latent_channels))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        shape = [1, -1] + [1] * (x.ndim - 2)
        return (x - self.get_buffer("mean-of-means").view(*shape).to(x)) / \
            self.get_buffer("std-of-means").view(*shape).to(x)

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        shape = [1, -1] + [1] * (x.ndim - 2)
        return x * self.get_buffer("std-of-means").view(*shape).to(x) + \
            self.get_buffer("mean-of-means").view(*shape).to(x)

    @torch.no_grad()
    def set_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.get_buffer("mean-of-means").copy_(mean.flatten())
        self.get_buffer("std-of-means").copy_(std.flatten().clamp_min(1e-6))


# ---------------------------------------------------------------------------
# encoder / decoder block factories (LTX video_vae.py naming, mesh version)
# ---------------------------------------------------------------------------

# stride over (F, L)
_STRIDES = {"time": (2, 1), "mesh": (1, 2), "all": (2, 2)}

# default LTX-style stack: total 8x time, 32x mesh (patch 4x included).
# mirrors the documented LTX config: 1x space, 1x time, 2x all (+res/attn).
DEFAULT_ENCODER_BLOCKS_8x32: List[Tuple[str, dict]] = [
    ("res_x", {"num_layers": 2}),
    ("compress_mesh_res", {"multiplier": 2}),
    ("res_x", {"num_layers": 2}),
    ("compress_time_res", {"multiplier": 2}),
    ("res_x", {"num_layers": 2}),
    ("compress_all_res", {"multiplier": 2}),
    ("res_x", {"num_layers": 2}),
    ("compress_all_res", {"multiplier": 2}),
    ("res_x", {"num_layers": 2}),
    ("attn", {"num_heads": 8}),
]


def _time_factor_of(blocks: List[Tuple[str, dict]]) -> int:
    f = 1
    for name, _ in blocks:
        if name in ("compress_time", "compress_time_res", "compress_all", "compress_all_res"):
            f *= 2
    return f


def _mesh_factor_of(blocks: List[Tuple[str, dict]], patch_size: int) -> int:
    f = patch_size
    for name, _ in blocks:
        if name in ("compress_mesh", "compress_mesh_res", "compress_all", "compress_all_res"):
            f *= 2
    return f


def _make_encoder_block(name: str, cfg: dict, in_ch: int, norm_layer: str,
                        norm_groups: int, pad_mode: str) -> Tuple[nn.Module, int]:
    out_ch = in_ch
    if name == "res_x":
        block = UNetMidBlock2D(in_ch, num_layers=cfg.get("num_layers", 2),
                               resnet_groups=norm_groups, norm_layer=norm_layer,
                               spatial_padding_mode=pad_mode)
    elif name == "res_x_y":
        out_ch = in_ch * cfg.get("multiplier", 2)
        block = ResnetBlock2D(in_ch, out_ch, groups=norm_groups,
                              norm_layer=norm_layer, spatial_padding_mode=pad_mode)
    elif name in ("compress_time", "compress_mesh", "compress_all"):
        block = CausalConv2d(in_ch, out_ch, 3, stride=_STRIDES[name.split("_")[1] if name != "compress_all" else "all"],
                             spatial_padding_mode=pad_mode)
    elif name in ("compress_time_res", "compress_mesh_res", "compress_all_res"):
        out_ch = in_ch * cfg.get("multiplier", 2)
        key = name.replace("compress_", "").replace("_res", "")
        block = SpaceToDepthDownsample2D(in_ch, out_ch, stride=_STRIDES[key],
                                         spatial_padding_mode=pad_mode)
    elif name == "attn":
        block = AttnBlock2D(in_ch, num_heads=cfg.get("num_heads", 8),
                            qk_norm=cfg.get("qk_norm", True))
    else:
        raise ValueError(f"unknown encoder block {name!r}")
    return block, out_ch


def _decoder_bottleneck(base: int, blocks: List[Tuple[str, dict]]) -> int:
    mult = 1
    for name, params in blocks:
        cfg = {"num_layers": params} if isinstance(params, int) else params
        if name in ("compress_time", "compress_mesh", "compress_all"):
            mult *= cfg.get("multiplier", 1)
        elif name in ("compress_time_res", "compress_mesh_res", "compress_all_res", "res_x_y"):
            mult *= cfg.get("multiplier", 2)
    return base * mult


def _make_decoder_block(name: str, cfg: dict, in_ch: int, norm_layer: str,
                        norm_groups: int, pad_mode: str) -> Tuple[nn.Module, int]:
    out_ch = in_ch
    if name == "res_x":
        block = UNetMidBlock2D(in_ch, num_layers=cfg.get("num_layers", 2),
                               resnet_groups=norm_groups, norm_layer=norm_layer,
                               spatial_padding_mode=pad_mode)
    elif name == "res_x_y":
        out_ch = in_ch // cfg.get("multiplier", 2)
        block = ResnetBlock2D(in_ch, out_ch, groups=norm_groups,
                              norm_layer=norm_layer, spatial_padding_mode=pad_mode)
    elif name in ("compress_time", "compress_mesh", "compress_all"):
        out_ch = in_ch // cfg.get("multiplier", 1)
        key = name.split("_")[1] if name != "compress_all" else "all"
        block = DepthToSpaceUpsample2D(in_ch, stride=_STRIDES[key],
                                       out_channels_reduction_factor=cfg.get("multiplier", 1),
                                       residual=cfg.get("residual", False),
                                       spatial_padding_mode=pad_mode)
    elif name in ("compress_time_res", "compress_mesh_res", "compress_all_res"):
        # same upsampler as the plain variant; channel reduction via multiplier
        out_ch = in_ch // cfg.get("multiplier", 2)
        key = name.replace("compress_", "").replace("_res", "")
        block = DepthToSpaceUpsample2D(in_ch, stride=_STRIDES[key],
                                       out_channels_reduction_factor=cfg.get("multiplier", 2),
                                       residual=cfg.get("residual", False),
                                       spatial_padding_mode=pad_mode)
    elif name == "attn":
        block = AttnBlock2D(in_ch, num_heads=cfg.get("num_heads", 8),
                            qk_norm=cfg.get("qk_norm", True))
    else:
        raise ValueError(f"unknown decoder block {name!r}")
    return block, out_ch


class MeshEncoder(nn.Module):
    """LTX ``VideoEncoder`` for ``(B, C, F, L)`` inputs."""

    def __init__(self, in_channels: int = 3, base_channels: int = 128,
                 latent_channels: int = 128,
                 encoder_blocks: List[Tuple[str, dict]] = DEFAULT_ENCODER_BLOCKS_8x32,
                 patch_size: int = 4, norm_layer: str = "pixel_norm",
                 latent_log_var: str = "uniform",
                 spatial_padding_mode: str = "zeros", norm_groups: int = 32):
        super().__init__()
        assert latent_log_var in ("per_channel", "uniform", "constant", "none")
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.latent_log_var = latent_log_var
        self.time_factor = _time_factor_of(encoder_blocks)
        self.mesh_factor = _mesh_factor_of(encoder_blocks, patch_size)
        self.per_channel_statistics = PerChannelStatistics(latent_channels)

        self.conv_in = CausalConv2d(in_channels * patch_size, base_channels, 3,
                                    spatial_padding_mode=spatial_padding_mode)
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        for bname, bparams in encoder_blocks:
            cfg = {"num_layers": bparams} if isinstance(bparams, int) else dict(bparams)
            blk, ch = _make_encoder_block(bname, cfg, ch, norm_layer, norm_groups,
                                          spatial_padding_mode)
            self.down_blocks.append(blk)
        self.conv_norm_out = _make_norm(ch, norm_layer, norm_groups)
        self.conv_act = nn.SiLU()
        out_ch = latent_channels
        if latent_log_var == "per_channel":
            out_ch *= 2
        elif latent_log_var in ("uniform", "constant"):
            out_ch += 1
        self.conv_out = CausalConv2d(ch, out_ch, 3, spatial_padding_mode=spatial_padding_mode)

    def _crop_frames(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[2]
        if (n - 1) % self.time_factor != 0:
            crop = (n - 1) % self.time_factor
            logger.warning("Cropping last %d of %d frames to satisfy 1 + %d*k.",
                           crop, n, self.time_factor)
            x = x[:, :, :-crop, :]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw posterior params ``(B, C(+1|*2), F', L')`` (pre-normalisation)."""
        x = self._crop_frames(x)
        if x.shape[-1] % self.mesh_factor != 0:
            raise ValueError(
                f"mesh length {x.shape[-1]} not divisible by mesh factor "
                f"{self.mesh_factor}; pad it before encoding.")
        x = patchify_mesh(x, self.patch_size)
        x = self.conv_in(x)
        for blk in self.down_blocks:
            x = blk(x)
        x = self.conv_out(self.conv_act(self.conv_norm_out(x)))
        if self.latent_log_var == "uniform":
            means, logvar = x[:, :-1], x[:, -1:]
            logvar = logvar.repeat(1, means.shape[1], 1, 1)
            x = torch.cat([means, logvar], dim=1)
        elif self.latent_log_var == "constant":
            x = torch.cat([x[:, :-1], torch.full_like(x[:, :-1], -30.0)], dim=1)
        return x

    def latent_shape(self, frames: int, mesh_points: int) -> Tuple[int, int, int]:
        frames = frames - (frames - 1) % self.time_factor
        return (self.latent_channels, (frames - 1) // self.time_factor + 1,
                mesh_points // self.mesh_factor)


class MeshDecoder(nn.Module):
    """LTX ``ConvVideoDecoder`` for ``(B, Cl, F', L')`` latents."""

    def __init__(self, latent_channels: int = 128, out_channels: int = 3,
                 base_channels: int = 128,
                 decoder_blocks: List[Tuple[str, dict]] = DEFAULT_ENCODER_BLOCKS_8x32,
                 patch_size: int = 4, norm_layer: str = "pixel_norm",
                 causal: bool = False, spatial_padding_mode: str = "reflect",
                 norm_groups: int = 32):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.causal = causal
        self.per_channel_statistics = PerChannelStatistics(latent_channels)
        ch = _decoder_bottleneck(base_channels, decoder_blocks)
        self.conv_in = CausalConv2d(latent_channels, ch, 3,
                                    spatial_padding_mode=spatial_padding_mode)
        self.up_blocks = nn.ModuleList()
        for bname, bparams in reversed(decoder_blocks):
            cfg = {"num_layers": bparams} if isinstance(bparams, int) else dict(bparams)
            blk, ch = _make_decoder_block(bname, cfg, ch, norm_layer, norm_groups,
                                          spatial_padding_mode)
            self.up_blocks.append(blk)
        self.conv_norm_out = _make_norm(ch, norm_layer, norm_groups)
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv2d(ch, out_channels * patch_size, 3,
                                     spatial_padding_mode=spatial_padding_mode)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``(B, Cl, F', L') -> (B, C, F, L)`` (expects normalised latents)."""
        z = self.per_channel_statistics.un_normalize(z)
        h = self.conv_in(z, causal=self.causal)
        for blk in self.up_blocks:
            h = blk(h, causal=self.causal)
        h = self.conv_out(self.conv_act(self.conv_norm_out(h)), causal=self.causal)
        return unpatchify_mesh(h, self.patch_size, self.out_channels)


# ---------------------------------------------------------------------------
# VAE wrapper (same public API as AutoencoderKL1d in autoencoder.py)
# ---------------------------------------------------------------------------

class DiagonalGaussian2D:
    """Diagonal Gaussian over ``(B, C, F, L)`` (first half mean, second log-var)."""

    def __init__(self, params: torch.Tensor):
        self.mean, self.log_var = params.chunk(2, dim=1)
        self.log_var = self.log_var.clamp(-15.0, 15.0)
        self.std = (0.5 * self.log_var).exp()

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        return 0.5 * (self.mean.pow(2) + self.log_var.exp() - 1.0 - self.log_var).sum()


@dataclass
class AutoencoderKLLTXConfig:
    in_channels: int = 3
    base_channels: int = 128
    latent_channels: int = 128
    patch_size: int = 4
    encoder_blocks: List[Tuple[str, dict]] = field(
        default_factory=lambda: [tuple(b) for b in DEFAULT_ENCODER_BLOCKS_8x32])
    norm_layer: str = "pixel_norm"
    latent_log_var: str = "uniform"   # LTX-2 default; 'per_channel' | 'constant' | 'none'
    decoder_causal: bool = False      # LTX conv-decoder default
    attn_heads: Optional[int] = None  # override every attn block's num_heads
    dropout: float = 0.0
    kl_weight: float = 1e-6


class AutoencoderKLLTX(nn.Module):
    """KL-VAE with LTX-2 style encoder/decoder for ``(B, C, F, L)`` data.

    Mirrors the ``AutoencoderKL1d`` API so ``TrainerVAE1D`` and the latent
    utilities from ``train_vae_cylinder.py`` work unchanged::

        vae = AutoencoderKLLTX(cfg)
        posterior = vae.encode(x)          # DiagonalGaussian2D
        z = posterior.sample()             # or .mode()
        x_hat = vae.decode(z_normalised)
        x_hat, posterior = vae(x)
        loss, rec, kl = vae.loss(x, x_hat, posterior)
    """

    def __init__(self, cfg: AutoencoderKLLTXConfig):
        super().__init__()
        self.cfg = cfg
        blocks = [tuple(b) for b in cfg.encoder_blocks]
        if cfg.attn_heads is not None:
            blocks = [(n, {**(p if isinstance(p, dict) else {"num_layers": p}),
                            "num_heads": cfg.attn_heads} if n == "attn" else (n, p))
                      for n, p in blocks]
        self.register_buffer("scale_factor", torch.tensor(1.0))
        self.encoder = MeshEncoder(cfg.in_channels, cfg.base_channels,
                                   cfg.latent_channels, blocks, cfg.patch_size,
                                   cfg.norm_layer, cfg.latent_log_var)
        # share the statistics module so fitted stats stay consistent
        self.decoder = MeshDecoder(cfg.latent_channels, cfg.in_channels,
                                   cfg.base_channels, blocks, cfg.patch_size,
                                   cfg.norm_layer, causal=cfg.decoder_causal)
        self.time_factor = self.encoder.time_factor
        self.mesh_factor = self.encoder.mesh_factor

    # -- public API ---------------------------------------------------------
    def encode(self, x: torch.Tensor) -> DiagonalGaussian2D:
        params = self.encoder(x)
        if self.cfg.latent_log_var == "none":
            params = torch.cat([params, torch.full_like(params, -30.0)], dim=1)
        return DiagonalGaussian2D(params)

    def encode_latent(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Normalised latent ready for diffusion training / saving."""
        post = self.encode(x)
        z = post.mode() if deterministic else post.sample()
        return self.encoder.per_channel_statistics.normalize(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, sample_posterior: bool = True):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(self.encoder.per_channel_statistics.normalize(z)), posterior

    def loss(self, x: torch.Tensor, x_hat: torch.Tensor, posterior: DiagonalGaussian2D):
        rec = F.mse_loss(x_hat, x)
        kl = posterior.kl() / x.numel()
        return rec + self.cfg.kl_weight * kl, rec, kl

    # -- helpers ------------------------------------------------------------
    @torch.no_grad()
    def set_scale_factor(self, loader, device: str = "cuda", num_batches: int = 16):
        self.eval()
        stds = []
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            stds.append(self.encode(x).mode().std().item())
        self.scale_factor = torch.tensor(sum(stds) / len(stds),
                                         device=self.scale_factor.device)
        print(f"[AutoencoderKLLTX] scale_factor set to {self.scale_factor.item():.4f}")

    def latent_shape(self, frames: int, mesh_points: int) -> Tuple[int, int, int]:
        return self.encoder.latent_shape(frames, mesh_points)

    def parameter_count(self) -> str:
        return f"{sum(p.numel() for p in self.parameters()) / 1e6:.2f}M"

    def sync_statistics(self) -> None:
        """Copy encoder statistics buffers into the decoder (keep after fitting)."""
        self.decoder.per_channel_statistics.set_statistics(
            self.encoder.per_channel_statistics.get_buffer("mean-of-means").clone(),
            self.encoder.per_channel_statistics.get_buffer("std-of-means").clone())


def denormalise_latents(z: torch.Tensor, stats_path: str) -> torch.Tensor:
    """Apply before passing diffusion samples to ``vae.decode()`` (cylinder-style)."""
    import numpy as np
    stats = np.load(stats_path)
    mean = torch.from_numpy(stats["mean"]).to(z)
    std = torch.from_numpy(stats["std"]).to(z)
    return z * std + mean
