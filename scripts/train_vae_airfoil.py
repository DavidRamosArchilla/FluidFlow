from data.load_airfoil_unsteady import load_airfoil_unsteady, load_meta
from fluidFlow.video_vae import AutoencoderKLLTX, AutoencoderKLLTXConfig
from fluidFlow.trainer_vae import TrainerVAE
from fluidFlow.evaluation import AirfoilUnsteadyEvaluator, AirfoilUnsteadyFFTEvaluator

import os
import shutil

import numpy as np
import torch
import torch._dynamo  # noqa: F401  (must stay BEFORE any tensorflow import below:
# torch optimizers lazily import torch._dynamo at construction time, and that
# import segfaults once TF's native libs are loaded — pre-importing avoids it)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)

# you need first to download the data, see data/airfoil_unsteady/download_data.sh:
# download_data.sh airfoil <path>
# run this script from the repo root so the relative path below resolves
data_dir = "data/airfoil_unsteady/airfoil"

# ---------- data config ----------
channels = 4               # 3->[u,v,p]  4->[u,v,p,rho]
# LTX causal time compression needs F = 1 + 8*k -> 297 = 1 + 8*37 or 601 = 1 + 8*75
time_frames = 601
time_mode = "uniform"      # "uniform" or "first"
max_train_samples = 32     # None for full 1000
max_valid_samples = 8      # None for full 100
max_test_samples = 8       # None for full 100
# max_train_samples = None  
# max_valid_samples = None   
# max_test_samples = None  
# VAE size: LTX-faithful would be base=128/latent=128 (very heavy);
# start small, raise latent_channels toward 128 once it reconstructs well.
base_channels = 64
latent_channels = 32
patch_size = 1
kl_weight = 1e-6
train_batch_size = 16
gradient_accumulate_every = 1
train_lr = 1e-4
train_steps = 70000
results_folder = 'results/airfoil_unsteady/first_4_channels'
# ---------- evaluation config ----------
eval_space = "physical"  # "physical", "normalized" or "both"
eval_sample = "auto"     # GIF sample: "auto", "auto:K" or int
eval_fft_samples = "auto"  # FFT samples: "auto", "auto:K" or "i,j,k"
# -----------------------------------

# mesh must be divisible by 32 (LTX total mesh factor incl. patchify),
# so pad to a multiple of 32 here (the loader does the padding).
dataset_train, dataset_valid, dataset_test, coefficients = load_airfoil_unsteady(
    data_dir,
    channels=channels,
    time_frames=time_frames,
    time_mode=time_mode,
    max_train_samples=max_train_samples,
    max_valid_samples=max_valid_samples,
    max_test_samples=max_test_samples,
    patch_size=32,
)

# VAE works on (N, C, F, L)
fields_train = dataset_train.tensors[0].permute(0, 2, 1, 3).contiguous()
fields_valid = dataset_valid.tensors[0].permute(0, 2, 1, 3).contiguous()
fields_test = dataset_test.tensors[0].permute(0, 2, 1, 3).contiguous()
print("transposed:", tuple(fields_train.shape), "(N,C,F,L)")

# frames must satisfy F = 1 + 8*k (LTX causal time compression)
n_frames = fields_train.shape[2]
if (n_frames - 1) % 8 != 0:
    keep = n_frames - (n_frames - 1) % 8
    print(f"Cropping frames {n_frames} -> {keep} to satisfy 1 + 8*k")
    fields_train = fields_train[:, :, :keep, :]
    fields_valid = fields_valid[:, :, :keep, :]
    fields_test = fields_test[:, :, :keep, :]
print("final train shape", tuple(fields_train.shape))

train_dataset = TensorDataset(fields_train)
valid_dataset = TensorDataset(fields_valid)
test_dataset = TensorDataset(fields_test)

cfg = AutoencoderKLLTXConfig(
    in_channels=channels,
    base_channels=base_channels,
    latent_channels=latent_channels,
    kl_weight=kl_weight,
    patch_size=patch_size,
)
model = AutoencoderKLLTX(cfg)
print("parameters:", model.parameter_count())
print("latent shape per sample:", model.latent_shape(fields_train.shape[2], fields_train.shape[3]),
      "(C_lat, F', L')")

trainer = TrainerVAE(
    model,
    train_dataset,
    dataset_test=valid_dataset,
    train_batch_size=train_batch_size,
    gradient_accumulate_every=gradient_accumulate_every,
    split_batches=True,
    train_num_steps=train_steps,
    results_folder=results_folder,
    save_every=10000,
    train_lr=train_lr,
    amp=True,
    mixed_precision_type='bf16',
    max_grad_norm=1.0,
    compile_model=True,
    eta_min_scheduler=1e-6,
)

# trainer.train()
trainer.load(os.path.join(results_folder, "vae_best.pt"))
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
os.makedirs(results_folder, exist_ok=True)
# norm stats in (1, C, 1, 1) VAE layout (loader stats are (1, 1, C, 1))
fields_mean = coefficients["fields_mean"].permute(0, 2, 1, 3).contiguous()
fields_std = coefficients["fields_std"].permute(0, 2, 1, 3).contiguous()
torch.save({"fields_mean": fields_mean, "fields_std": fields_std,
            "target_length": coefficients["target_length"],
            "original_length": coefficients["original_length"],
            "frames": fields_train.shape[2]},
           os.path.join(results_folder, "norm_stats.pt"))


@torch.no_grad()
def reconstruct_test(vae, dataset, batch_size=2, device="cuda", num_workers=4):
    """Deterministic VAE reconstructions of a dataset, (N, C, F, L) on CPU."""
    vae = vae.to(device)
    vae.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    recs = []
    for batch in tqdm(loader, desc="Reconstructing"):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x_hat, _ = vae(x.to(device), sample_posterior=False)
        recs.append(x_hat.cpu().float())
    return torch.cat(recs, dim=0)


@torch.no_grad()
def extract_latents(vae, dataset, batch_size=8, device="cuda",
                    num_workers=4, desc="Extracting latents"):
    """Encode a dataset to normalised latents via the posterior mean.

    Returns float32 tensor of shape (N, latent_channels, F_down, L_down).
    """
    vae.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        pin_memory=True, num_workers=num_workers)
    latents = []
    for batch in tqdm(loader, desc=desc):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        latents.append(vae.encode_latent(x.to(device), deterministic=True).cpu().float())
    return torch.cat(latents, dim=0)


@torch.inference_mode()
def save_latents(vae, dataset_train, dataset_test, save_dir,
                 batch_size=8, device="cuda", num_workers=4):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    z_train = extract_latents(vae, dataset_train, batch_size, device,
                              num_workers, desc="Train latents")
    z_test = extract_latents(vae, dataset_test, batch_size, device,
                             num_workers, desc="Test latents")

    print(f"Train latents: {tuple(z_train.shape)} (N, C_lat, F', L')")
    print(f"Test  latents: {tuple(z_test.shape)}")

    # statistics from train set only, per channel; keepdim -> (1, C_lat, 1, 1)
    mean = z_train.mean(dim=(0, 2, 3), keepdim=True)
    std = z_train.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)

    print("\nPer-channel stats (train):")
    for c in range(mean.shape[1]):
        print(f"  ch {c}:  mean={mean[0, c, 0, 0]:.4f}  std={std[0, c, 0, 0]:.4f}")

    z_train_norm = (z_train - mean) / std
    z_test_norm = (z_test - mean) / std  # use TRAIN stats on test

    print("\nAfter standardisation:")
    print(f"  train — mean: {z_train_norm.mean():.6f}  std: {z_train_norm.std():.6f}")
    print(f"  test  — mean: {z_test_norm.mean():.6f}   std: {z_test_norm.std():.6f}")

    np.save(save_dir / "latents_train.npy", z_train_norm.numpy())
    np.save(save_dir / "latents_test.npy", z_test_norm.numpy())
    np.savez(save_dir / "latents_stats.npz", mean=mean.numpy(), std=std.numpy())
    print(f"\nSaved to {save_dir}/")


if trainer.accelerator.is_main_process:
    vae = trainer._unwrapped_model()
    recs = reconstruct_test(vae, test_dataset, batch_size=1, device=trainer.device)
    torch.save(recs, os.path.join(results_folder, "reconstructions.pt"))

    # ---- final evaluation, same as train_airfoil_unsteady.py (physical units) ----
    # back to (N, F, C, L) loader layout, trim mesh padding, denormalize
    n_channels = channels
    channel_names = ["u", "v", "p", "rho"][:n_channels]
    orig_len = coefficients["original_length"]
    fmean = coefficients["fields_mean"].cpu()
    fstd = coefficients["fields_std"].cpu()
    recs_denorm = recs.permute(0, 2, 1, 3)[..., :orig_len] * fstd + fmean
    test_data = dataset_test.tensors[0].cpu()[..., :orig_len]
    test_data_denorm = test_data * fstd + fmean
    conds_denorm = (dataset_test.tensors[1].cpu() * coefficients["conds_std"].cpu()
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
        test_data_denorm, recs_denorm,
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
    fft_eval(test_data_denorm, recs_denorm, conds=conds_denorm, mesh_pos=mesh_pos)
    fft_eval.print_metrics()

    save_latents(vae, train_dataset, test_dataset,
                 save_dir=results_folder, batch_size=2, device=trainer.device)
