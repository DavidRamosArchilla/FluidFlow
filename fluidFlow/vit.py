"""Autoregressive ViT for one-step prediction of 1D-mesh fields.

Reuses the :class:`fluidFlow.dit.DiT` backbone (blocks, conditioning, Muon
param split) but overrides ``forward`` / ``sample`` so it can be trained with
the same :class:`fluidFlow.trainer.Trainer`:

    - ``context`` is the current state u_t, shape (B, C, L)
    - ``x`` is the expected output u_{t+1}, shape (B, C, L)

With ``residual=True`` the network predicts the normalized increment
(u_{t+1} - u_t - delta_mean) / delta_std, so the output starts as the identity
map (the final layer is zero-initialised). Conditions (Mach, alpha) enter
through adaLN as in the DiT; there is no diffusion timestep.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .dit import DiT, FinalLayer1D


def fourier_encode(coords, num_frequencies=16):
    """
    coords: (N, d) expected in [-1, 1]
    out: (N, d * 2 * num_frequencies)
    """
    freqs = 2.0 ** torch.arange(num_frequencies, device=coords.device)  # (F,)
    x = coords.unsqueeze(-1) * freqs * np.pi  # (N, d, F)
    enc = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # (N, d, 2F)
    return enc.flatten(-2)  # (N, d * 2F)


class CoordEmbedder(nn.Module):
    """Fourier features of the node coordinates -> MLP -> (N, D)."""
    def __init__(self, embed_dim, coord_dim=2, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.mlp = nn.Sequential(
            nn.Linear(coord_dim * 2 * num_frequencies, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, coords):  # coords: (N, d)
        return self.mlp(fourier_encode(coords, self.num_frequencies))


class ViT(DiT):
    """
    One-step autoregressive transformer: u_t -> u_{t+1}.

    Args (on top of the DiT ones):
        out_channels: predicted channels (default: in_channels).
        residual: predict the (normalized) increment instead of u_{t+1}.
        coords: optional (L, d) node coordinates. If given, a Fourier
            coordinate embedding replaces the sin-cos index embedding; with
            patch_size > 1 it is averaged over the nodes of each patch.
        num_frequencies: Fourier frequencies of the coordinate embedding.
        valid_length: number of real nodes; nodes beyond it are padding,
            excluded from the loss and kept at zero during rollouts.
    """
    def __init__(self, out_channels=None, residual=True, coords=None, num_frequencies=16,
                 valid_length=None, **kwargs):
        kwargs.setdefault("class_dropout_prob", 0.0)  # deterministic regression, no CFG
        super().__init__(**kwargs)
        assert self.is_1d, "ViT only supports 1D (mesh) inputs"
        self.residual = residual
        self.is_video = False
        self.out_channels = self.channels if out_channels is None else out_channels
        hidden_size = self.hidden_size
        input_size = self.x_embedder.seq_len
        self.final_layer = FinalLayer1D(hidden_size, self.patch_size, self.out_channels,
                                        bias=kwargs.get("use_bias", True))
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # no diffusion timestep: drop the (unused) timestep embedder so that
        # DDP does not complain about parameters without gradients
        del self.t_embedder
        if self.y_embedder.dropout_prob == 0:
            del self.y_embedder.null_classes_emb  # never used without condition dropout

        self.use_coord_pe = coords is not None
        if self.use_coord_pe:
            coords = torch.as_tensor(np.asarray(coords), dtype=torch.float32)
            assert coords.ndim == 2 and coords.shape[0] <= input_size, \
                f"coords must be (L, d) with L <= {input_size}, got {tuple(coords.shape)}"
            # scale to [-1, 1] and pad like the fields
            coords = coords / coords.abs().max().clamp(min=1e-12)
            coords = torch.nn.functional.pad(coords, (0, 0, 0, input_size - coords.shape[0]))
            self.register_buffer("coords", coords)
            self.coord_pe = CoordEmbedder(hidden_size, coord_dim=coords.shape[1],
                                          num_frequencies=num_frequencies)

        # node mask for padded meshes: (1, 1, L)
        node_mask = torch.ones(1, 1, input_size)
        if valid_length is not None:
            node_mask[..., valid_length:] = 0
        self.register_buffer("node_mask", node_mask)

        # increment normalization stats (set with set_delta_stats)
        self.register_buffer("delta_mean", torch.zeros(1, self.out_channels, 1))
        self.register_buffer("delta_std", torch.ones(1, self.out_channels, 1))

    def set_delta_stats(self, mean, std):
        """mean / std: per-channel stats of u_{t+1} - u_t, any shape with C elements."""
        self.delta_mean.copy_(torch.as_tensor(mean, dtype=torch.float32).reshape(1, -1, 1))
        self.delta_std.copy_(torch.as_tensor(std, dtype=torch.float32).reshape(1, -1, 1))

    def _pos_embed(self):
        if not self.use_coord_pe:
            return self.pos_embed  # (1, S, D)
        emb = self.coord_pe(self.coords)  # (L, D)
        emb = emb.reshape(-1, self.patch_size, emb.shape[-1]).mean(dim=1)  # (S, D)
        return emb.unsqueeze(0)

    def predict(self, context, classes, **kwargs):
        """Raw network output for the current state context (B, C, L)."""
        x = self.x_embedder(context) + self._pos_embed()  # (B, S, D)
        force_drop_ids = kwargs.get("force_drop_ids", None)
        c = self.y_embedder(classes, self.training, force_drop_ids)  # (B, D)
        for block in self.blocks:
            if self.gradient_checkpointing:
                x = checkpoint(block, x, c, self.feat_rope, use_reentrant=False)
            else:
                x = block(x, c, self.feat_rope)  # (B, S, D)
        x = self.final_layer(x, c)  # (B, S, patch_size * out_channels)
        return self.unpatchify(x)  # (B, out_channels, L)

    def step(self, context, classes, **kwargs):
        """One autoregressive step u_t -> u_{t+1} (in float32)."""
        out = self.predict(context, classes, **kwargs).float()
        if self.residual:
            out = context[:, :self.out_channels].float() + out * self.delta_std + self.delta_mean
        return out * self.node_mask

    def forward(self, x, classes=None, context=None, return_loss=True, **kwargs):
        """
        x: (B, C, L) next state u_{t+1} (target)
        context: (B, C, L) current state u_t
        """
        if not return_loss:
            return self.step(context, classes, **kwargs)
        out = self.predict(context, classes, **kwargs).float()
        target = x.float()
        if self.residual:
            target = (target - context[:, :self.out_channels].float() - self.delta_mean) / self.delta_std
        err = (target - out) ** 2 * self.node_mask
        return err.sum() / (self.node_mask.sum() * err.shape[0] * err.shape[1])

    def sample(self, classes, context=None, num_steps=None, **model_kwargs):
        """
        num_steps=None: one-step prediction, returns (B, C, L).
        num_steps=K: autoregressive rollout from context, returns
            (B, K + 1, C, L) with the initial state as the first frame.
        """
        self.eval()
        with torch.inference_mode():
            if num_steps is None:
                return self.step(context, classes, **model_kwargs)
            state = context.float() * self.node_mask
            frames = [state]
            for _ in range(num_steps):
                state = self.step(state, classes, **model_kwargs)
                frames.append(state)
        return torch.stack(frames, dim=1)
