"""Autoregressive ViT on the cylinder dataset (data/cylinder).

Same pipeline as scripts/train_airfoil_unsteady_ar.py, without the VAE: the
ViT is trained on one-step pairs (u_t -> u_{t+1}) with teacher forcing and
predicts the normalized increment (residual). It is conditioned on the case
parameters in constants.h5 (cases/Re by default, see cond_keys), so new cases
with other Re / viscosity values only need to be dropped into the split
folders as more <i>.h5 files. At the end the EMA model is rolled out from the
first frame of every test trajectory and scored with the RegressionEvaluator
in physical units.

Run from the repo root:
    python scripts/train_cylinder_ar.py
    accelerate launch --num_processes 4 scripts/train_cylinder_ar.py
"""
from data.load_airfoil_unsteady import make_pair_dataset, make_rollout_dataset, compute_delta_stats
from data.load_cylinder import load_cylinder
from fluidFlow.vit import ViT
from fluidFlow.trainer import Trainer
from fluidFlow.evaluation import RegressionEvaluator

import json
import os
import shutil

import numpy as np
import torch


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)

# run this script from the repo root so the relative path below resolves
data_dir = "data/cylinder"

# ---------- data config ----------
cond_keys = ("Re",)       # datasets under cases/ in constants.h5 used as conditions
time_stride = 1           # keep every k-th snapshot (larger AR step)
max_train_samples = None
max_valid_samples = None
max_test_samples = None
valid_frame_stride = 5    # validation pairs: every k-th starting frame (one-step MSE during training)
# ---------- model config ----------
patch_size = 8            # 8000 cells -> 1000 tokens
depth = 8
hidden_size = 512
num_heads = 8             # head_dim 64
mlp_ratio = 4
residual = True           # predict normalized u_{t+1} - u_t instead of u_{t+1}
use_coord_pe = True       # Fourier embedding of the cell centers instead of sin-cos index
num_frequencies = 16
# ---------- training config ----------
train_batch_size = 16
gradient_accumulate_every = 1
train_lr = 2e-4           # AdamW lr (1D params: biases, norms, patch embedder)
train_steps = 2000       # optimizer steps (each = gradient_accumulate_every micro-batches)
muon_lr = 2e-4
muon_adjust_lr_fn = "match_rms_adamw"
muon_weight_decay = 1e-2
ema_decay = 0.999
save_and_sample_every = 2000
results_folder = 'results/cylinder/vit_ar_p8_d8_h512_first_test'
load_milestone = None     # set to m to skip training and evaluate model-{m}.pt
# ---------- evaluation config ----------
eval_batch_size = 8
# -----------------------------------

dataset_train, dataset_valid, dataset_test, coefficients = load_cylinder(
    data_dir,
    cond_keys=cond_keys,
    time_stride=time_stride,
    max_train_samples=max_train_samples,
    max_valid_samples=max_valid_samples,
    max_test_samples=max_test_samples,
    patch_size=patch_size,
)

fields_shape = dataset_train.tensors[0].shape  # (N, F, C, L)
print("Train fields shape", fields_shape)
valid_length = coefficients["original_length"]  # real cells (rest is padding)
n_ar_steps = fields_shape[1] - 1

# one-step increment stats (normalize the residual target)
delta_mean, delta_std = compute_delta_stats(dataset_train.tensors[0], valid_length=valid_length)
print("delta mean per channel", delta_mean.tolist())
print("delta std per channel", delta_std.tolist())

# (u_{t+1}, conds, u_t) pairs for training / validation
pairs_train = make_pair_dataset(dataset_train)
pairs_valid = make_pair_dataset(dataset_valid, frame_stride=valid_frame_stride)
print(f"AR pairs: train {len(pairs_train)}, valid {len(pairs_valid)}; rollout steps {n_ar_steps}")

model = ViT(
    depth=depth,
    hidden_size=hidden_size,
    patch_size=patch_size,
    num_heads=num_heads,
    input_size=fields_shape[-1],
    cond_dim=dataset_train.tensors[1].shape[1],  # len(cond_keys)
    class_dropout_prob=0.0,
    in_channels=fields_shape[2],
    out_channels=fields_shape[2],
    learn_sigma=False,
    use_swiglu=True,
    qk_norm=True,  # to avoid stability issues with bf16
    attn_type="vanilla",
    mlp_ratio=mlp_ratio,
    residual=residual,
    coords=coefficients["centers"] if use_coord_pe else None,
    num_frequencies=num_frequencies,
    valid_length=valid_length,
)
# set before the Trainer is built so the EMA copy carries the stats too
model.set_delta_stats(delta_mean, delta_std)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

trainer = Trainer(
    model,
    dataset=pairs_train,
    dataset_test=pairs_valid,  # one-step validation MSE during training
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
    split_batches=True,
    compile_model=True,
)

if trainer.accelerator.is_main_process:
    shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
    # save norm stats for denormalization at eval
    torch.save(
        {
            "fields_mean": coefficients["fields_mean"],
            "fields_std": coefficients["fields_std"],
            "conds_mean": coefficients["conds_mean"],
            "conds_std": coefficients["conds_std"],
            "delta_mean": delta_mean,
            "delta_std": delta_std,
            "target_length": coefficients["target_length"],
            "original_length": coefficients["original_length"],
            "channel_names": coefficients["channel_names"],
            "cond_names": coefficients["cond_names"],
        },
        os.path.join(results_folder, "norm_stats.pt"),
    )

if load_milestone is None:
    trainer.train()
else:
    trainer.load(load_milestone)

# Autoregressive rollout on the test set from the first frame of each trajectory
trainer.ema.ema_model.eval()
rollout_test = make_rollout_dataset(dataset_test)  # (traj, conds, traj[:, 0])
samples, _ = trainer.eval_model(rollout_test, batch_size=eval_batch_size, use_autocast=True, num_steps=n_ar_steps)

if trainer.accelerator.is_main_process:
    # ---- final evaluation (physical units, (N, F, C, L) fields) ----
    orig_len = coefficients["original_length"]
    fmean = coefficients["fields_mean"].cpu()
    fstd = coefficients["fields_std"].cpu()
    samples_denorm = samples.cpu()[..., :orig_len] * fstd + fmean
    test_data_denorm = dataset_test.tensors[0][..., :orig_len] * fstd + fmean
    torch.save(samples_denorm, os.path.join(results_folder, "test_predictions_ema.pt"))

    # frame 0 is the given initial state, so only the predicted frames are scored
    y_true, y_pred = test_data_denorm[:, 1:], samples_denorm[:, 1:]
    evaluator = RegressionEvaluator()
    metrics = {"all": evaluator(y_true, y_pred)}
    print("\nRollout, all channels:")
    evaluator.print_metrics()
    for c, name in enumerate(coefficients["channel_names"]):
        metrics[name] = evaluator(y_true[:, :, c], y_pred[:, :, c])
        print(f"\nRollout, channel {name}:")
        evaluator.print_metrics()

    # relative L2 error of each predicted frame, averaged over the test cases
    err = torch.linalg.vector_norm(y_pred - y_true, dim=(2, 3)) / torch.linalg.vector_norm(y_true, dim=(2, 3))
    metrics["rel_l2_per_step"] = err.mean(0).tolist()
    print("\nrelative L2 at steps", {k: round(metrics["rel_l2_per_step"][k - 1], 5)
                                     for k in (1, 10, 50, n_ar_steps) if k <= n_ar_steps})

    metrics = {k: {m: float(v) for m, v in d.items()} if isinstance(d, dict) else d for k, d in metrics.items()}
    with open(os.path.join(results_folder, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved predictions and metrics to {results_folder}")
