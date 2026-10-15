"""ViT regressor from point coordinates to the fields on those points.

Maps a surface mesh / point cloud ``context`` (B, 3, L) to fields (B, C, L),
e.g. the surface pressure of the DrivAerNet cars. Consecutive points are
grouped into patches of ``patch_size`` (sort them spatially first, see
``data.load_drivaernet.morton_order``), so each token sees a small piece of
the surface and the transformer mixes them globally.

Meshes with different numbers of points are padded to a common length;
``mask`` (B, L) marks the real points: padded tokens are excluded from
attention and padded points from the loss. There is no conditioning, so the
blocks are plain pre-norm transformer blocks (no adaLN).

The interface matches :class:`fluidFlow.trainer.Trainer` with dict batches
``{"x": fields, "context": coords, "mask": mask}``.
"""

import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp
from torch.utils.checkpoint import checkpoint

from .attention import Attention
from .basic_modules import SwiGLUFFN
from .vit import fourier_encode


class Block(nn.Module):
    """Pre-norm transformer block with a key padding mask."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, bias=True, use_swiglu=True, qk_norm=True):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=bias)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=bias, proj_bias=bias, qk_norm=qk_norm)
        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=bias)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim), bias=bias)
        else:
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim,
                           act_layer=lambda: nn.GELU(approximate="tanh"), drop=0, bias=bias)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x


class PointViT(nn.Module):
    """
    Args:
        in_channels: point coordinate channels (3 for xyz).
        out_channels: predicted field channels.
        patch_size: consecutive points per token.
        num_frequencies: Fourier features of the (normalized) coordinates
            added to the raw ones before patching (0 = raw coordinates only).
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        patch_size=16,
        hidden_size=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        use_bias=True,
        use_swiglu=True,
        qk_norm=True,
        num_frequencies=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = out_channels  # Trainer interface
        self.cond_dim = None  # no conditioning
        self.patch_size = patch_size
        self.num_frequencies = num_frequencies
        point_dim = in_channels * (1 + 2 * num_frequencies)
        self.x_embedder = nn.Conv1d(point_dim, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_ratio=mlp_ratio, bias=use_bias,
                  use_swiglu=use_swiglu, qk_norm=qk_norm)
            for _ in range(depth)
        ])
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=use_bias)
        self.final_layer = nn.Linear(hidden_size, patch_size * out_channels, bias=use_bias)
        self.initialize_weights()
        self.gradient_checkpointing = False

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # patch embedder like nn.Linear; output starts at zero (the field mean)
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)
        nn.init.constant_(self.final_layer.weight, 0)
        if self.final_layer.bias is not None:
            nn.init.constant_(self.final_layer.bias, 0)

    def embed_points(self, coords):
        """(B, d, L) coordinates -> (B, d * (1 + 2F), L) per-point features."""
        if self.num_frequencies == 0:
            return coords
        enc = fourier_encode(coords.transpose(1, 2), self.num_frequencies)  # (B, L, d * 2F)
        return torch.cat([coords, enc.transpose(1, 2)], dim=1)

    def predict(self, context, mask=None):
        """context: (B, in_channels, L) coordinates, L a multiple of patch_size.
        Returns the (B, out_channels, L) fields in float32, zero on padded points."""
        B, _, L = context.shape
        p = self.patch_size
        assert L % p == 0, f"number of points {L} must be a multiple of patch_size {p}"
        x = self.x_embedder(self.embed_points(context)).transpose(1, 2)  # (B, S, D)
        # a token is real if any of its points is
        token_mask = None if mask is None else mask.view(B, L // p, p).any(dim=-1)  # (B, S)
        for block in self.blocks:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                x = checkpoint(block, x, token_mask, use_reentrant=False)
            else:
                x = block(x, token_mask)
        x = self.final_layer(self.norm_final(x))  # (B, S, p * C)
        x = x.reshape(B, L // p, p, self.out_channels).permute(0, 3, 1, 2)  # (B, C, S, p)
        x = x.reshape(B, self.out_channels, L).float()
        if mask is not None:
            # where, not a product: NaN * 0 would keep garbage from padded rows
            x = torch.where(mask.unsqueeze(1), x, 0.0)
        return x

    def forward(self, x, context, mask=None, return_loss=True, **kwargs):
        """
        x: (B, C, L) target fields
        context: (B, in_channels, L) point coordinates
        mask: (B, L) bool, True on real points
        Returns the MSE over the real points.
        """
        pred = self.predict(context, mask)
        if not return_loss:
            return pred
        err = (pred - x.float()) ** 2
        if mask is None:
            return err.mean()
        return torch.where(mask.unsqueeze(1), err, 0.0).sum() / (mask.sum() * err.shape[1])

    def sample(self, context, mask=None, **kwargs):
        self.eval()
        with torch.inference_mode():
            return self.predict(context, mask)

    def get_2d_params(self):
        """Parameters for Muon (weight matrices)."""
        return [p for p in self.parameters() if p.dim() == 2]

    def get_1d_params(self):
        """Parameters for AdamW (biases, norms, conv patch embedder)."""
        return [p for p in self.parameters() if p.dim() != 2]

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
