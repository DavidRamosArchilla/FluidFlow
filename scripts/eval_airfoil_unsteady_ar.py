"""Evaluate saved milestones of an autoregressive ViT run with full rollouts.

Rolls out the EMA weights of every requested milestone (model-{m}.pt) from
the first frame of each trajectory of the test (or valid) split and reports
the momentum RMSE (MeshGraphNets-style, as in AirfoilUnsteadyEvaluator), the
global physical relative L2 and the error per horizon, so runs and
checkpoints can be compared on the same footing. Milestone 0 is the run's
init_checkpoint (the model before fine-tuning).

With --full_eval the predictions are saved and the full evaluators
(AirfoilUnsteadyEvaluator + AirfoilUnsteadyFFTEvaluator, with GIFs and plots)
are run for each milestone, into {out_dir}/m{m}/.

The model config is read from the training script copied into the results
folder, so the same command works for any AR / unroll run. Run from the repo
root; milestones are split across processes, e.g. on 8 GPUs:

    accelerate launch --num_processes 8 scripts/eval_airfoil_unsteady_ar.py \
        --results_folder results/airfoil_unsteady/vit_ar_d10_h1024_unroll_ft --milestones all

Outputs {results_folder}/eval_milestones_{split}/m{m}.npz and summary.json.
"""
import argparse
import ast
import glob
import json
import os
import re

import numpy as np
import torch
from accelerate import PartialState

from data.load_airfoil_unsteady import load_airfoil_split, load_meta
from fluidFlow.evaluation import AirfoilUnsteadyEvaluator, AirfoilUnsteadyFFTEvaluator
from fluidFlow.evaluation.airfoil_unsteady import airfoil_box
from fluidFlow.vit import ViT

HORIZONS = [(1, 50), (50, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 601)]
CHANNEL_NAMES = ["u", "v", "p", "rho"]


def read_config(results_folder):
    """Top-level literal assignments of the training script copied into the results folder."""
    scripts = glob.glob(os.path.join(results_folder, "train_airfoil_unsteady_ar*.py"))
    assert len(scripts) == 1, f"expected one training script in {results_folder}, found {scripts}"
    cfg = {}
    for node in ast.parse(open(scripts[0]).read()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                cfg[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return cfg


def load_ema_weights(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    ema_state = {k[len("ema_model."):]: v for k, v in ckpt["ema"].items() if k.startswith("ema_model.")}
    model.load_state_dict(ema_state)
    return ckpt["step"]


def momentum_sq_err(pred, gt, node_mask=None):
    """(B, F) squared error of (rho*u, rho*v), mean over both components and the nodes; physical inputs."""
    iu, iv, ir = (CHANNEL_NAMES.index(k) for k in ("u", "v", "rho"))
    if node_mask is not None:
        pred, gt = pred[..., node_mask], gt[..., node_mask]
    m_p = pred[:, :, [iu, iv]] * pred[:, :, ir:ir + 1]
    m_g = gt[:, :, [iu, iv]] * gt[:, :, ir:ir + 1]
    return (m_p - m_g).pow(2).mean((2, 3))


@torch.no_grad()
def rollout_errors(model, fields, conds, fields_mean, fields_std, batch_size, device,
                   use_autocast=True, roi_mask=None, keep_predictions=False):
    """
    Roll out every trajectory from its first frame and accumulate the errors:
        mse: (N, F, C) MSE in normalized space
        gt_energy: (C,) sum of the squared physical GT (for the relative L2)
        mom: (N, F) momentum squared error, whole mesh; mom_roi: same within roi_mask
        preds: (N, F, C, L) normalized predictions on the CPU (only with keep_predictions)
    """
    n, n_frames, n_channels, _ = fields.shape
    out = {"mse": torch.zeros(n, n_frames, n_channels, dtype=torch.float64),
           "gt_energy": torch.zeros(n_channels, dtype=torch.float64),
           "mom": torch.zeros(n, n_frames, dtype=torch.float64),
           "mom_roi": torch.zeros(n, n_frames, dtype=torch.float64)}
    preds = [] if keep_predictions else None
    has_momentum = n_channels == len(CHANNEL_NAMES)
    for a in range(0, n, batch_size):
        gt = fields[a:a + batch_size].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
            pred = model.sample(classes=conds[a:a + batch_size].to(device), context=gt[:, 0],
                                num_steps=n_frames - 1).float()
        out["mse"][a:a + batch_size] = (pred - gt).pow(2).mean(-1).double().cpu()
        gt_phys = gt.double() * fields_std + fields_mean
        pred_phys = pred.double() * fields_std + fields_mean
        out["gt_energy"] += gt_phys.pow(2).sum((0, 1, 3)).cpu()
        if has_momentum:
            out["mom"][a:a + batch_size] = momentum_sq_err(pred_phys, gt_phys).cpu()
            if roi_mask is not None:
                out["mom_roi"][a:a + batch_size] = momentum_sq_err(pred_phys, gt_phys, roi_mask).cpu()
        if keep_predictions:
            preds.append(pred.cpu())
        del gt, pred, gt_phys, pred_phys
    return out, (torch.cat(preds) if keep_predictions else None)


def summarize(errors, fields_std, n_nodes):
    """Momentum RMSE (as momentum_rollout_rmse), global physical relative L2 and horizon-binned normalized MSE."""
    mse = torch.as_tensor(errors["mse"])
    var = torch.as_tensor(fields_std, dtype=torch.float64).reshape(-1) ** 2
    sse_phys = (mse * var).sum((0, 1)) * n_nodes  # per channel
    gt_energy = torch.as_tensor(errors["gt_energy"])
    out = {}
    # initial frame excluded, mean over frames per trajectory, then over trajectories, then sqrt
    for key in ("mom", "mom_roi"):
        mom = torch.as_tensor(errors[key])[:, 1:]
        if mom.abs().sum() == 0:
            continue
        name = "momentum_rmse" if key == "mom" else "momentum_rmse_roi"
        out[f"{name}_all"] = float(mom.mean(1).mean().sqrt())
        if mom.shape[1] > 50:
            out[f"{name}_50"] = float(mom[:, :50].mean(1).mean().sqrt())
    out["l2_phys_global"] = float((sse_phys.sum() / gt_energy.sum()).sqrt())
    out["l2_phys_per_channel"] = (sse_phys / gt_energy).sqrt().tolist()
    out["mse_norm_all"] = float(mse[:, 1:].mean())
    for a, b in HORIZONS:
        if a < mse.shape[1]:
            out[f"mse_norm_{a}_{b}"] = float(mse[:, a:b].mean())
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_folder", required=True)
    parser.add_argument("--milestones", default="all", help='"all", or comma separated (0 = init_checkpoint)')
    parser.add_argument("--split", default="test", choices=["test", "valid"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=None, help="rollout steps (default: whole trajectory)")
    parser.add_argument("--fp32", action="store_true",
                        help="roll out in fp32 without flash attention (slower), to check the effect of bf16")
    parser.add_argument("--full_eval", action="store_true",
                        help="save the predictions and run the full field + FFT evaluators per milestone")
    parser.add_argument("--data_dir", default="data/airfoil_unsteady/airfoil")
    args = parser.parse_args()

    state = PartialState()
    cfg = read_config(args.results_folder)
    assert not cfg.get("use_vae", False), "VAE runs are not supported here"
    assert cfg.get("time_frames", 601) in (None, 601), "only full-resolution runs are supported here"

    if args.milestones == "all":
        found = [int(re.search(r"model-(\d+)\.pt$", p).group(1))
                 for p in glob.glob(os.path.join(args.results_folder, "model-*.pt"))]
        milestones = sorted(found) + ([0] if cfg.get("init_checkpoint") else [])
    else:
        milestones = [int(m) for m in args.milestones.split(",")]
    milestones = sorted(set(milestones))
    if args.fp32:
        import fluidFlow.attention
        fluidFlow.attention.is_flash_attn_available = False  # FA4 only supports bf16/fp16
    tag = args.split + (f"_{args.num_steps}steps" if args.num_steps else "") + ("_fp32" if args.fp32 else "")
    out_dir = os.path.join(args.results_folder, f"eval_milestones_{tag}")
    os.makedirs(out_dir, exist_ok=True)

    # only the evaluated split, normalized with the run's train stats
    stats = torch.load(os.path.join(args.results_folder, "norm_stats.pt"), weights_only=False)
    fields, conds_phys, mesh_pos, cells, node_type = load_airfoil_split(
        args.split, base_dir=args.data_dir, channels=cfg["channels"], max_samples=args.max_samples,
        return_node_type=True)
    fields_mean = stats["fields_mean"].reshape(1, 1, -1, 1).float()
    fields_std = stats["fields_std"].reshape(1, 1, -1, 1).float()
    fields = (fields - fields_mean) / fields_std
    conds = (conds_phys - stats["conds_mean"]) / stats["conds_std"]
    if args.num_steps is not None:
        fields = fields[:, :args.num_steps + 1]
    n_nodes = fields.shape[-1]
    assert n_nodes == stats["original_length"] == stats["target_length"], "padded runs are not supported here"
    mesh_pos, cells, node_type = np.asarray(mesh_pos)[0], np.asarray(cells)[0], np.asarray(node_type)
    # same region of interest around the wing as AirfoilUnsteadyEvaluator(roi="auto")
    roi = airfoil_box(mesh_pos, node_type)
    roi_mask = torch.from_numpy((mesh_pos[:, 0] >= roi[0]) & (mesh_pos[:, 0] <= roi[1])
                                & (mesh_pos[:, 1] >= roi[2]) & (mesh_pos[:, 1] <= roi[3])).to(state.device)

    model = ViT(
        depth=cfg["depth"],
        hidden_size=cfg["hidden_size"],
        patch_size=cfg["patch_size"],
        num_heads=cfg["num_heads"],
        input_size=n_nodes,
        cond_dim=conds.shape[1],
        class_dropout_prob=0.0,
        in_channels=cfg["channels"],
        out_channels=cfg["channels"],
        learn_sigma=False,
        use_swiglu=True,
        qk_norm=True,
        attn_type="vanilla",
        mlp_ratio=cfg["mlp_ratio"],
        residual=cfg["residual"],
        coords=mesh_pos if cfg["use_coord_pe"] else None,
        num_frequencies=cfg["num_frequencies"],
        valid_length=n_nodes,
    ).to(state.device).eval()

    for m in milestones[state.process_index::state.num_processes]:
        path = cfg["init_checkpoint"] if m == 0 else os.path.join(args.results_folder, f"model-{m}.pt")
        step = load_ema_weights(model, path)
        errors, preds = rollout_errors(model, fields, conds, fields_mean.to(state.device), fields_std.to(state.device),
                                       args.batch_size, state.device, use_autocast=not args.fp32,
                                       roi_mask=roi_mask, keep_predictions=args.full_eval)
        np.savez(os.path.join(out_dir, f"m{m}.npz"), **{k: v.numpy() for k, v in errors.items()},
                 step=step, checkpoint=path)
        print(f"[rank {state.process_index}] milestone {m} (step {step}): "
              f"{summarize(errors, stats['fields_std'], n_nodes)}", flush=True)

        if args.full_eval:
            m_dir = os.path.join(out_dir, f"m{m}")
            os.makedirs(m_dir, exist_ok=True)
            torch.save(preds, os.path.join(m_dir, "test_predictions_ema.pt"))  # normalized, as the training script
            channel_names = CHANNEL_NAMES[:cfg["channels"]]
            pred_phys = preds * fields_std + fields_mean
            gt_phys = fields * fields_std + fields_mean
            del preds
            fields_eval = AirfoilUnsteadyEvaluator(out_dir=os.path.join(m_dir, "eval_fields"),
                                                   channel_names=channel_names, space="physical", sample_idx="auto")
            fields_eval(gt_phys, pred_phys, mesh_pos=mesh_pos, node_type=node_type, cells=cells,
                        conds=conds_phys, norm_stats={"fields_mean": fields_mean, "fields_std": fields_std})
            meta = load_meta(os.path.join(args.data_dir, "meta.json"))
            fft_eval = AirfoilUnsteadyFFTEvaluator(out_dir=os.path.join(m_dir, "eval_fft"),
                                                   dt=float(meta.get("dt", 0.0002)),
                                                   channel_names=channel_names, samples="auto")
            fft_eval(gt_phys, pred_phys, conds=conds_phys, mesh_pos=mesh_pos)
            del pred_phys, gt_phys

    state.wait_for_everyone()
    if state.is_main_process:
        summary = {}
        for m in milestones:
            z = np.load(os.path.join(out_dir, f"m{m}.npz"))
            summary[m] = {"step": int(z["step"]), "checkpoint": str(z["checkpoint"]),
                          **summarize(z, stats["fields_std"], n_nodes)}
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        first = next(iter(summary.values()))
        cols = [c for c in first if c not in ("step", "checkpoint", "l2_phys_per_channel")]
        print("\nmilestone    step  " + "  ".join(c.replace("mse_norm_", "mse_") for c in cols))
        for m, s in summary.items():
            print(f"{m:9d}  {s['step']:6d}  " + "  ".join(f"{s[c]:.4g}" for c in cols))
        print(f"Saved {out_dir}/summary.json")
