"""Adapter that lets FluidFlow models be scored by FluidBench (github.com/sp-droid/FluidBench).

``fluidbench.run(model, config)`` only needs two methods on the model:

    predict(loader)          one step per test pair (teacher forcing) -> (N, C, L)
    predict_rollout(loader)  rollout from the first item of a batch_size=1 loader -> (T, C, L)

Batches are ((scalars (B, S), fields (B, C, L)), target), in physical units
and in the benchmark's channel order (e.g. [p, Ux, Uy]). :class:`FluidBenchWrapper`
turns them into the normalized, reordered and padded states the FluidFlow
models are trained on, and maps the predictions back. Only the one-step map
(:meth:`FluidBenchWrapper._step`) is model specific: the AR ViT uses
``ViT.step``; a diffusion model would override it with its sampler.

This module does not import fluidbench itself.
"""

import ast
import glob
import os
import re
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# benchmark variable names -> FluidFlow channel names
_CHANNEL_ALIASES = {"ux": "u", "uy": "v", "uz": "w", "pressure": "p"}


def _canonical(name):
    name = name.casefold()
    return _CHANNEL_ALIASES.get(name, name)


def read_training_config(results_folder, pattern="train_*.py"):
    """Top-level literal assignments of the training script copied into the results folder."""
    scripts = glob.glob(os.path.join(results_folder, pattern))
    assert len(scripts) == 1, f"expected one {pattern} in {results_folder}, found {scripts}"
    cfg = {}
    for node in ast.parse(open(scripts[0]).read()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                cfg[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return cfg


def latest_milestone(results_folder):
    milestones = [int(m.group(1)) for path in glob.glob(os.path.join(results_folder, "model-*.pt"))
                  if (m := re.fullmatch(r"model-(\d+)\.pt", os.path.basename(path)))]
    assert milestones, f"no model-*.pt in {results_folder}"
    return max(milestones)


class FluidBenchWrapper(nn.Module):
    """
    FluidBench interface around a model working on normalized (B, C, L) states.

    Args:
        model: FluidFlow model with ``step(context, classes)`` (u_t -> u_{t+1},
            normalized); ``init_context`` / ``advance_context`` are used when present.
        fields_mean, fields_std: per-channel stats in the model's channel order.
        conds_mean, conds_std: stats of the model conditions.
        target_length: model input length (mesh points + padding).
        original_length: real mesh points.
        channel_order: model channel c is benchmark channel channel_order[c].
        cond_indices: model condition j is benchmark scalar cond_indices[j].
        autocast_dtype: mixed precision on GPU (None = float32). CPU always runs in float32.
    """
    def __init__(self, model, fields_mean, fields_std, conds_mean, conds_std, target_length, original_length,
                 channel_order, cond_indices, autocast_dtype=torch.bfloat16):
        super().__init__()
        self.model = model
        self.target_length = target_length
        self.original_length = original_length
        self.channel_order = list(channel_order)
        self.inverse_order = np.argsort(self.channel_order).tolist()
        self.cond_indices = list(cond_indices)
        self.autocast_dtype = autocast_dtype
        self.register_buffer("fields_mean", torch.as_tensor(fields_mean, dtype=torch.float32).reshape(1, -1, 1))
        self.register_buffer("fields_std", torch.as_tensor(fields_std, dtype=torch.float32).reshape(1, -1, 1))
        self.register_buffer("conds_mean", torch.as_tensor(conds_mean, dtype=torch.float32).reshape(1, -1))
        self.register_buffer("conds_std", torch.as_tensor(conds_std, dtype=torch.float32).reshape(1, -1))

    # ---------- conversions ----------
    def _normalize(self, fields):
        """Benchmark fields (B, C, L) -> normalized model state (B, C, target_length)."""
        x = fields.to(self.fields_mean.device, torch.float32)[:, self.channel_order]
        x = (x - self.fields_mean) / self.fields_std
        return F.pad(x, (0, self.target_length - x.shape[-1]))

    def _denormalize(self, state):
        """Normalized model state (B, C, target_length) -> benchmark fields (B, C, L)."""
        x = state.float()[..., :self.original_length] * self.fields_std + self.fields_mean
        return x[:, self.inverse_order]

    def _conds(self, scalars):
        c = scalars.to(self.conds_mean.device, torch.float32).reshape(len(scalars), -1)[:, self.cond_indices]
        return (c - self.conds_mean) / self.conds_std

    def _autocast(self):
        device = self.fields_mean.device
        if self.autocast_dtype is None or device.type == "cpu":
            return nullcontext()
        return torch.autocast(device.type, dtype=self.autocast_dtype)

    # ---------- model hooks ----------
    def _init_context(self, state):
        init = getattr(self.model, "init_context", None)
        return state if init is None else init(state)

    def _advance_context(self, context, state):
        advance = getattr(self.model, "advance_context", None)
        return state if advance is None else advance(context, state)

    def _step(self, context, conds):
        """One normalized step u_t -> u_{t+1}. Override for models with another update (e.g. a sampler)."""
        return self.model.step(context, conds)

    # ---------- FluidBench interface ----------
    @torch.inference_mode()
    def predict(self, loader):
        """One-step predictions for every (u_t, u_{t+1}) pair of the loader: (N, C, L) on CPU."""
        self.eval()
        predictions = []
        for (scalars, fields), _ in loader:
            with self._autocast():
                state = self._step(self._init_context(self._normalize(fields)), self._conds(scalars))
            predictions.append(self._denormalize(state).cpu())
        return torch.cat(predictions)

    @torch.inference_mode()
    def predict_rollout(self, loader):
        """Rollout from the fields of the first item; one step per item, with that item's scalars: (T, C, L)."""
        self.eval()
        batches = list(loader)
        assert all(len(fields) == 1 for (_, fields), _ in batches), "predict_rollout expects batch_size=1"
        (_, first_fields), _ = batches[0]
        context = self._init_context(self._normalize(first_fields))
        frames = []
        for (scalars, _), _ in batches:  # the later fields are ground truth and are never used
            with self._autocast():
                state = self._step(context, self._conds(scalars))
            frames.append(self._denormalize(state)[0].cpu())
            context = self._advance_context(context, state)
        return torch.stack(frames)

    # ---------- constructors ----------
    @classmethod
    def from_ar_vit_results(cls, results_folder, benchmark_channels, benchmark_scalars, milestone=None,
                            script_pattern="train_cylinder_ar*.py", **kwargs):
        """AR ViT from a results folder of scripts/train_cylinder_ar.py (EMA weights of model-{milestone}.pt).

        benchmark_channels / benchmark_scalars: the benchmark's flowVariables and
        inputScalarNames, used to map its channels and scalars onto the model's.
        The checkpoint path, step and loss history are kept in ``wrapper.checkpoint``.
        """
        from data.load_cylinder import load_cylinder_mesh
        from ..vit import ViT

        cfg = read_training_config(results_folder, script_pattern)
        stats = torch.load(os.path.join(results_folder, "norm_stats.pt"), map_location="cpu", weights_only=True)
        milestone = latest_milestone(results_folder) if milestone is None else milestone

        model_channels = list(stats["channel_names"])
        bench_channels = [_canonical(name) for name in benchmark_channels]
        missing = [name for name in model_channels if name not in bench_channels]
        assert not missing and len(model_channels) == len(bench_channels), \
            f"model channels {model_channels} do not match benchmark channels {list(benchmark_channels)}"
        missing = [name for name in stats["cond_names"] if name not in benchmark_scalars]
        assert not missing, f"conditions {missing} are not benchmark scalars {list(benchmark_scalars)}"

        data_dir = cfg["data_dir"] if os.path.isabs(cfg["data_dir"]) else os.path.join(REPO_ROOT, cfg["data_dir"])
        coords = load_cylinder_mesh(os.path.join(data_dir, "train"))["centers"] if cfg["use_coord_pe"] else None
        n_channels = len(model_channels)
        # same arguments as scripts/train_cylinder_ar.py; strict loading below catches any drift
        model = ViT(
            depth=cfg["depth"],
            hidden_size=cfg["hidden_size"],
            patch_size=cfg["patch_size"],
            num_heads=cfg["num_heads"],
            input_size=stats["target_length"],
            cond_dim=len(stats["cond_names"]),
            class_dropout_prob=0.0,
            in_channels=n_channels,
            out_channels=n_channels,
            learn_sigma=False,
            use_swiglu=True,
            qk_norm=True,
            attn_type="vanilla",
            mlp_ratio=cfg["mlp_ratio"],
            residual=cfg["residual"],
            coords=coords,
            num_frequencies=cfg["num_frequencies"],
            valid_length=stats["original_length"],
        )
        path = os.path.join(results_folder, f"model-{milestone}.pt")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        ema_state = {k[len("ema_model."):]: v for k, v in ckpt["ema"].items() if k.startswith("ema_model.")}
        model.load_state_dict(ema_state)
        print(f"Loaded EMA weights from {path} (step {ckpt['step']})")

        wrapper = cls(
            model,
            fields_mean=stats["fields_mean"],
            fields_std=stats["fields_std"],
            conds_mean=stats["conds_mean"],
            conds_std=stats["conds_std"],
            target_length=stats["target_length"],
            original_length=stats["original_length"],
            channel_order=[bench_channels.index(name) for name in model_channels],
            cond_indices=[list(benchmark_scalars).index(name) for name in stats["cond_names"]],
            **kwargs,
        )
        wrapper.checkpoint = {"path": path, "step": int(ckpt["step"]), "loss_history": ckpt["loss_history"]}
        return wrapper
