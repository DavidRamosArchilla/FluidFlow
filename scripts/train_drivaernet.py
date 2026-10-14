"""ViT regression of the DrivAerNet++ surface pressure from the point coordinates.

The model (fluidFlow.point_vit.PointViT) receives the xyz coordinates of every
surface point of a car and predicts the pressure on each of them. Cars have
different numbers of points (~360k-560k), so batches are padded by
``drivaernet_collate_fn`` and a point mask is passed to the model: padded
tokens are excluded from attention and padded points from the loss/metrics.

The points of each car are sorted along a Morton curve when the cache is
built, so each patch of ``patch_size`` consecutive points is a small piece of
surface. The VTK meshes are read once into the cache (see
data/load_drivaernet.py); after that only the cache is needed.
"""
from data.load_drivaernet import load_drivaernet, drivaernet_collate_fn, index_vtk_files, write_prediction_vtk
from fluidFlow.point_vit import PointViT
from fluidFlow.trainer import Trainer
from fluidFlow.evaluation import RegressionEvaluator

import csv
import json
import os
import shutil
from functools import partial

import numpy as np
import torch
from torch.utils.data import Subset


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)

# run this script from the repo root so the relative paths below resolve
# ---------- data config ----------
cache_path = "data/driveaernet/drivaernet_cache.pt"
vtk_dir = "/home/d.ramos/DrivAerNet/PressureVTK"  # needed to build the cache and to export VTKs
split_dir = "/home/d.ramos/DrivAerNet/splits"      # needed to build the cache
field_names = ("p",)
max_valid_samples = 200   # validation cars evaluated at every milestone (None = all 1148)
# ---------- model config ----------
patch_size = 16           # consecutive (Morton-sorted) points per token: ~28k tokens per car
depth = 12
hidden_size = 512
num_heads = 8             # head_dim 64 (FA4 needs 32/64/128)
mlp_ratio = 4
num_frequencies = 6       # Fourier features of xyz added before patching (0 = raw xyz)
# ---------- training config ----------
train_batch_size = 8      # global batch, split over the GPUs
gradient_accumulate_every = 1
gradient_checkpointing = False  # ~45 GB per 4 cars without it on one GPU
train_lr = 2e-4           # AdamW lr (1D params: biases, norms, patch embedder)
train_steps = 100000
muon_lr = 2e-4
muon_adjust_lr_fn = "match_rms_adamw"
muon_weight_decay = 1e-2
ema_decay = 0.999
save_and_sample_every = 10000
results_folder = f"results/drivaernet/point_vit_p{patch_size}_d{depth}_h{hidden_size}"
# ---------- evaluation config ----------
eval_batch_size = 8
export_vtk = True         # save the best / median / worst test cars with predicted pressure and error
# -----------------------------------

dataset_train, dataset_valid, dataset_test, coefficients = load_drivaernet(
    cache_path, vtk_dir=vtk_dir, split_dir=split_dir, field_names=field_names)
if max_valid_samples is not None and len(dataset_valid) > max_valid_samples:
    dataset_valid = Subset(dataset_valid, np.linspace(0, len(dataset_valid) - 1, max_valid_samples).astype(int))
n_channels = len(coefficients["fields_mean"])

# pad every batch to the longest car, rounded up to a whole patch
collate_fn = partial(drivaernet_collate_fn, pad_multiple=patch_size)

model = PointViT(
    in_channels=3,  # xyz
    out_channels=n_channels,
    patch_size=patch_size,
    hidden_size=hidden_size,
    depth=depth,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio,
    use_swiglu=True,
    qk_norm=True,  # to avoid stability issues with bf16
    num_frequencies=num_frequencies,
)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

trainer = Trainer(
    model,
    dataset=dataset_train,
    dataset_test=dataset_valid,  # masked validation MSE at every milestone
    train_batch_size=train_batch_size,
    train_lr=train_lr,
    train_num_steps=train_steps,
    gradient_accumulate_every=gradient_accumulate_every,
    ema_decay=ema_decay,
    amp=True,
    mixed_precision_type='bf16',
    results_folder=results_folder,
    save_and_sample_every=save_and_sample_every,
    eta_min_scheduler=4e-6,
    max_grad_norm=1.0,
    use_muon=True,
    muon_lr=muon_lr,
    muon_adjust_lr_fn=muon_adjust_lr_fn,
    muon_weight_decay=muon_weight_decay,
    gradient_checkpointing=gradient_checkpointing,
    split_batches=True,
    dataloader_collate_fn=collate_fn,
    save_eval_samples=False,  # validation predictions are ~0.5 GB per milestone
)

if trainer.accelerator.is_main_process:
    shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
    # save norm stats for denormalization at eval (standalone or below)
    torch.save(coefficients, os.path.join(results_folder, "norm_stats.pt"))

trainer.train()
# trainer.load(10)

# ---- prediction on the test set (multi-GPU if available) ----
trainer.ema.ema_model.eval()
pred_file = os.path.join(results_folder, "test_predictions_ema.pt")
if os.path.exists(pred_file):
    preds = torch.load(pred_file)  # list of (C, N_i) in physical units
else:
    samples, _, mask = trainer.eval_model(dataset_test, batch_size=eval_batch_size, use_autocast=True,
                                          return_mask=True)
    if trainer.accelerator.is_main_process:
        fmean = coefficients["fields_mean"][:, None]
        fstd = coefficients["fields_std"][:, None]
        lengths = mask.sum(dim=1).tolist()
        preds = [samples[i, :, :n] * fstd + fmean for i, n in enumerate(lengths)]
        torch.save(preds, pred_file)

if trainer.accelerator.is_main_process:
    # ---- final evaluation (physical units, real points only) ----
    stats = coefficients
    trues = []
    for i in range(len(dataset_test)):
        f, _ = dataset_test[i]
        trues.append(f * stats["fields_std"][:, None] + stats["fields_mean"][:, None])
    assert all(p.shape == t.shape for p, t in zip(preds, trues)), "predictions do not match the test cars"

    # vector fields (e.g. wall shear stress) span several channels
    channel_names = list(field_names) if n_channels == len(field_names) else [f"ch{c}" for c in range(n_channels)]
    metrics = {}
    per_car = {"name": dataset_test.names}
    for c, name in enumerate(channel_names):
        # global metrics over all the test points
        evaluator = RegressionEvaluator()
        global_metrics = evaluator(torch.cat([t[c] for t in trues]), torch.cat([p[c] for p in preds]))
        print(f"\n===== {name} =====")
        evaluator.print_metrics()
        # per-car relative errors (mean over cars, as usually reported on DrivAerNet++)
        rel_l2 = np.array([((p[c] - t[c]).norm() / t[c].norm()).item() for p, t in zip(preds, trues)])
        rel_l1 = np.array([((p[c] - t[c]).abs().sum() / t[c].abs().sum()).item() for p, t in zip(preds, trues)])
        mae = np.array([(p[c] - t[c]).abs().mean().item() for p, t in zip(preds, trues)])
        print(f"per-car rel L2: mean {rel_l2.mean():.4f}  median {np.median(rel_l2):.4f}  max {rel_l2.max():.4f}")
        print(f"per-car rel L1: mean {rel_l1.mean():.4f}")
        metrics[name] = {**{k: float(v) for k, v in global_metrics.items()},
                         "rel_l2_mean": float(rel_l2.mean()), "rel_l2_median": float(np.median(rel_l2)),
                         "rel_l1_mean": float(rel_l1.mean())}
        per_car[f"rel_l2_{name}"] = rel_l2
        per_car[f"rel_l1_{name}"] = rel_l1
        per_car[f"mae_{name}"] = mae

    with open(os.path.join(results_folder, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(results_folder, "per_car_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(per_car.keys())
        writer.writerows(zip(*[v if k == "name" else [f"{x:.6g}" for x in v] for k, v in per_car.items()]))

    if export_vtk:
        # best / median / worst test cars by rel L2 of the first field, viewable in ParaView
        out_dir = os.path.join(results_folder, "eval_vtk")
        os.makedirs(out_dir, exist_ok=True)
        vtk_paths = index_vtk_files(vtk_dir)
        order = np.argsort(per_car[f"rel_l2_{channel_names[0]}"])
        for tag, i in [("best", order[0]), ("median", order[len(order) // 2]), ("worst", order[-1])]:
            name = dataset_test.names[i]
            out_path = os.path.join(out_dir, f"{tag}_{name}.vtk")
            write_prediction_vtk(vtk_paths[name], out_path, preds[i].numpy(), field_names=field_names)
            print(f"Saved {out_path}")
