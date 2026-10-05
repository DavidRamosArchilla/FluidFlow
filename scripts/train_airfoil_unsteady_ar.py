"""Autoregressive ViT on the unsteady airfoil dataset.

The ViT is trained on one-step pairs (u_t -> u_{t+1}) with teacher forcing
and predicts the normalized increment (residual). At test time it is rolled
out from the first frame of each test trajectory and evaluated with the same
evaluators as the diffusion model (scripts/train_airfoil_unsteady.py).

With use_vae=True the same pipeline runs on the precomputed VAE latents
(N, C_lat, F', L'): pairs are (z_t -> z_{t+1}) in latent frames, the rollout
starts from the encoded first latent frame and is decoded before evaluation.
"""
from data.load_airfoil_unsteady import (
    load_airfoil_unsteady, load_meta, make_pair_dataset, make_rollout_dataset, compute_delta_stats,
)
from fluidFlow.vit import ViT
from fluidFlow.video_vae import AutoencoderKLLTX, AutoencoderKLLTXConfig
from fluidFlow.trainer import Trainer
from fluidFlow.trainer_vae import denormalise_latents, decode_latents
from fluidFlow.evaluation import AirfoilUnsteadyEvaluator, AirfoilUnsteadyFFTEvaluator

import os
import shutil

import numpy as np
import torch
from torch.utils.data import TensorDataset


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)

# you need first to download the data, see data/airfoil_unsteady/download_data.sh:
# download_data.sh airfoil <path>
# run this script from the repo root so the relative path below resolves
data_dir = "data/airfoil_unsteady/airfoil"

# ---------- data config ----------
channels = 4              # 3->[u,v,p]  4->[u,v,p,rho]
time_frames = 601         # temporal subsampling (None for the full 601 frames); fewer frames = larger AR step
time_mode = "uniform"     # "uniform" or "first"
max_train_samples = None
max_valid_samples = None
max_test_samples = None
valid_frame_stride = 20   # validation pairs: every k-th starting frame (one-step MSE during training)
# ---------- model config ----------
patch_size = 1
depth = 8
hidden_size = 512
num_heads = 8             # head_dim 64 (FA4 needs 32/64/128)
mlp_ratio = 2.5
residual = True           # predict normalized u_{t+1} - u_t instead of u_{t+1}
use_coord_pe = True       # Fourier embedding of the mesh coordinates instead of sin-cos node index
num_frequencies = 16
# ---------- training config ----------
train_batch_size = 16
gradient_accumulate_every = 1
train_lr = 2e-4           # AdamW lr (1D params: biases, norms, patch embedder)
train_steps = 100000      # optimizer steps (each = gradient_accumulate_every micro-batches)
muon_lr = 2e-4
muon_adjust_lr_fn = "match_rms_adamw"
muon_weight_decay = 1e-2
ema_decay = 0.999
results_folder = 'results/airfoil_unsteady/vit_ar_d8_h512'
# ---------- evaluation config ----------
eval_batch_size = 8
eval_space = "physical"  # "physical", "normalized" or "both"
eval_sample = "auto"     # GIF sample: "auto", "auto:K" or int
eval_fft_samples = "auto"  # FFT samples: "auto", "auto:K" or "i,j,k"
# ---------- vae / latent config ----------
use_vae = False  # set True to train the ViT on precomputed VAE latents
vae_dir = 'results/airfoil_unsteady/vae_32_channels'
vae_checkpoint = 'vae_best.pt'
vae_base_channels = 64   # must match the trained VAE
vae_latent_channels = 32  # must match the trained VAE
vae_patch_size = 1       # must match the trained VAE
latent_patch_size = 1    # ViT patch over the latent mesh dim (must divide L')
# -----------------------------------

# the VAE was trained on fields padded to a multiple of 32, so the
# precomputed latents only match if we load the fields with the same padding
dataset_train, dataset_valid, dataset_test, coefficients = load_airfoil_unsteady(
    data_dir,
    channels=channels,
    time_frames=time_frames,
    time_mode=time_mode,
    max_train_samples=max_train_samples,
    max_valid_samples=max_valid_samples,
    max_test_samples=max_test_samples,
    patch_size=32 if use_vae else patch_size,
)

fields_shape = dataset_train.tensors[0].shape  # (N, F, C, Nnodes)
print("Train fields shape", fields_shape)

# GT fields are always needed for the final evaluation; in VAE mode the
# AR datasets below hold latents instead, so keep the fields aside.
fields_test_dataset = dataset_test

vit_patch_size = patch_size
valid_length = coefficients["original_length"]  # real mesh nodes (rest is padding)
coords = np.asarray(coefficients["mesh_pos"])[0] if use_coord_pe else None  # (L, 2)
keep_frames = None  # VAE mode: frames kept after the 1+8k crop (None = all)
vae = None
if use_vae:
    vae = AutoencoderKLLTX(AutoencoderKLLTXConfig(
        in_channels=channels,
        base_channels=vae_base_channels,
        latent_channels=vae_latent_channels,
        patch_size=vae_patch_size,
    ))

    # same frame crop the VAE was trained with (F = 1 + 8*k)
    n_full_frames = fields_shape[1]
    keep_frames = n_full_frames if (n_full_frames - 1) % 8 == 0 else n_full_frames - (n_full_frames - 1) % 8
    if keep_frames != n_full_frames:
        print(f"Cropping frames {n_full_frames} -> {keep_frames} to match the VAE latents")

    # No TrainerVAE here: a second accelerator breaks the Trainer.
    # The checkpoint was saved from a compiled model (TrainerVAE with
    # compile_model=True), so compile first — its keys carry the "_orig_mod." prefix.
    vae = torch.compile(vae)
    ckpt = torch.load(os.path.join(vae_dir, vae_checkpoint), map_location="cpu")
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    print(f"Loaded VAE weights from {os.path.join(vae_dir, vae_checkpoint)}")

    # precomputed latents are (N, C_lat, F', L'): transpose vs the (N, F, C, L) fields
    z_train = torch.from_numpy(np.load(os.path.join(vae_dir, "latents_train.npy"))).float()
    z_valid = torch.from_numpy(np.load(os.path.join(vae_dir, "latents_valid.npy"))).float()
    z_test = torch.from_numpy(np.load(os.path.join(vae_dir, "latents_test.npy"))).float()
    print("Precomputed latents:", tuple(z_train.shape), tuple(z_valid.shape), tuple(z_test.shape), "(N, C_lat, F', L')")

    def _to_frame_latents(z, conds, name):
        assert z.shape[0] == conds.shape[0], \
            f"{name}: {z.shape[0]} latents vs {conds.shape[0]} conditions — " \
            "the precomputed latents must come from the same splits/caps"
        return TensorDataset(z.permute(0, 2, 1, 3).contiguous(), conds)  # (N, F', C_lat, L')

    dataset_train = _to_frame_latents(z_train, dataset_train.tensors[1], "train")
    dataset_valid = _to_frame_latents(z_valid, dataset_valid.tensors[1], "valid")
    dataset_test = _to_frame_latents(z_test, fields_test_dataset.tensors[1], "test")

    vit_patch_size = latent_patch_size
    valid_length = None  # latent mesh carries no padding
    coords = None        # latent nodes do not map to mesh coordinates
    print("Train latent shape (frame layout):", tuple(dataset_train.tensors[0].shape), "(N, F', C_lat, L')")

state_shape = dataset_train.tensors[0].shape  # (N, F, C, L) fields or latents
n_ar_steps = state_shape[1] - 1

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
    patch_size=vit_patch_size,
    num_heads=num_heads,
    input_size=state_shape[-1],
    cond_dim=dataset_train.tensors[1].shape[1],  # Mach, alpha
    class_dropout_prob=0.0,
    in_channels=state_shape[2],
    out_channels=state_shape[2],
    learn_sigma=False,
    use_swiglu=True,
    qk_norm=True,  # to avoid stability issues with bf16
    attn_type="vanilla",
    mlp_ratio=mlp_ratio,
    residual=residual,
    coords=coords,
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
    save_and_sample_every=20000,
    eta_min_scheduler=1e-6,
    max_grad_norm=1.0,
    use_muon=True,
    muon_lr=muon_lr,
    muon_adjust_lr_fn=muon_adjust_lr_fn,
    muon_weight_decay=muon_weight_decay,
    split_batches=True,
)

shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
# save norm stats for denormalization at eval (standalone or below)
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
    },
    os.path.join(results_folder, "norm_stats.pt"),
)

trainer.train()
# trainer.load(5)

# Autoregressive rollout on the test set from the first frame of each trajectory
trainer.ema.ema_model.eval()
rollout_test = make_rollout_dataset(dataset_test)  # (traj, conds, traj[:, 0])
pred_file = f"{results_folder}/latent_predictions.pt" if use_vae else f"{results_folder}/test_predictions_ema.pt"
if os.path.exists(pred_file):
    samples = torch.load(pred_file)
else:
    # this will roll out with multiple gpus, if available
    samples, seqs = trainer.eval_model(rollout_test, batch_size=eval_batch_size, use_autocast=True,
                                       num_steps=n_ar_steps)

if trainer.accelerator.is_main_process:
    # ---- final evaluation (physical units, (N, F, C, L) fields) ----
    n_channels = channels
    channel_names = ["u", "v", "p", "rho"][:n_channels]
    orig_len = coefficients["original_length"]
    fmean = coefficients["fields_mean"].cpu()
    fstd = coefficients["fields_std"].cpu()
    gt_fields = fields_test_dataset.tensors[0].cpu()
    if use_vae:
        torch.save(samples, pred_file)
        # latents (N, F', C_lat, L') -> VAE layout -> denormalize -> decode
        z = samples.cpu().permute(0, 2, 1, 3).contiguous()
        z = denormalise_latents(z, os.path.join(vae_dir, "latents_stats.npz"))
        decoded = decode_latents(vae, z, batch_size=2, device=trainer.device)  # (N, C, F, L)
        decoded = decoded.permute(0, 2, 1, 3)[..., :orig_len]
        torch.save(decoded, f"{results_folder}/test_predictions_ema.pt")
        samples_denorm = decoded * fstd + fmean
        test_data_denorm = gt_fields[:, :keep_frames][..., :orig_len] * fstd + fmean
    else:
        torch.save(samples, pred_file)
        samples_denorm = samples.cpu()[..., :orig_len] * fstd + fmean
        test_data_denorm = gt_fields[..., :orig_len] * fstd + fmean
    conds_denorm = (fields_test_dataset.tensors[1].cpu() * coefficients["conds_std"].cpu()
                    + coefficients["conds_mean"].cpu())
    mesh_pos = np.asarray(coefficients["mesh_pos"])[0]  # (L, 2)
    cells = np.asarray(coefficients["cells"])[0]  # (M, 3)
    node_type = np.asarray(coefficients["node_type"])  # (L,)

    fields_eval = AirfoilUnsteadyEvaluator(
        out_dir=os.path.join(results_folder, "eval_fields"),
        channel_names=channel_names,
        space=eval_space,
        sample_idx=eval_sample,
    )
    fields_eval(
        test_data_denorm, samples_denorm,
        mesh_pos=mesh_pos, node_type=node_type, cells=cells,
        conds=conds_denorm,
        norm_stats={"fields_mean": fmean, "fields_std": fstd},
    )
    fields_eval.print_metrics()

    meta = load_meta(os.path.join(data_dir, "meta.json"))
    fft_eval = AirfoilUnsteadyFFTEvaluator(
        out_dir=os.path.join(results_folder, "eval_fft"),
        dt=float(meta.get("dt", 0.0002)),
        channel_names=channel_names,
        samples=eval_fft_samples,
    )
    fft_eval(test_data_denorm, samples_denorm, conds=conds_denorm, mesh_pos=mesh_pos)
    fft_eval.print_metrics()
