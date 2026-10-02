from data.load_airfoil_unsteady import load_airfoil_unsteady, load_meta
from fluidFlow.video_dit import VideoDiT
from fluidFlow.video_vae import AutoencoderKLLTX, AutoencoderKLLTXConfig
from fluidFlow.trainer import Trainer
from fluidFlow.trainer_vae import denormalise_latents, decode_latents
from fluidFlow.flow_matching import create_flow_matching
from fluidFlow.evaluation import AirfoilUnsteadyEvaluator, AirfoilUnsteadyFFTEvaluator

import json
import os
import shutil

import numpy as np
import torch
from torch.utils.data import TensorDataset


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)

try:
    from fluidFlow.attention import is_flash_attn_available as _has_fa4
except Exception:
    _has_fa4 = False

# you need first to download the data, see data/airfoil_unsteady/download_data.sh:
# download_data.sh airfoil <path>
# run this script from the repo root so the relative path below resolves
data_dir = "data/airfoil_unsteady/airfoil"

# ---------- data config ----------
channels = 4              # 3->[u,v,p]  4->[u,v,p,rho]
time_frames = 601         # temporal subsampling (None for the full 601 frames)
time_mode = "uniform"     # "uniform" or "first"
max_train_samples = None
max_valid_samples = None
max_test_samples = None
# ---------- model config ----------
patch_size = 1
depth = 8
hidden_size = 1024
num_heads = 16
factorize = True          # True if time_frames > 32 to save memory
# ---------- training config ----------
train_lr = 2e-4           # AdamW lr (1D params: biases, norms, patch embedder)
train_steps = 100000      # optimizer steps (each = gradient_accumulate_every micro-batches)
# Muon (2D weights). "match_rms_adamw" scales the orthogonalised update to the
# AdamW update size, so muon_lr lives on the same scale as train_lr. With the
# default "original" scaling, lr=1e-4 gives ~30x smaller updates than AdamW.
muon_lr = 2e-4
muon_adjust_lr_fn = "match_rms_adamw"
muon_weight_decay = 1e-2
ema_decay = 0.999         # EMA horizon ~10k steps (0.995 -> ~2k steps)
use_lognorm = True        # logit-normal timestep sampling instead of uniform
results_folder = 'results/airfoil_unsteady/d8_p1_latent_v2'
# ---------- evaluation config ----------
eval_space = "physical"  # "physical", "normalized" or "both"
eval_sample = "auto"     # GIF sample: "auto", "auto:K" or int
eval_fft_samples = "auto"  # FFT samples: "auto", "auto:K" or "i,j,k"
# ---------- vae / latent-diffusion config ----------
use_vae = True  # set True to train the DiT on precomputed VAE latents
vae_dir = 'results/airfoil_unsteady/vae_32_channels'
vae_checkpoint = 'vae_best.pt'
vae_base_channels = 64   # must match the trained VAE
vae_latent_channels = 32  # must match the trained VAE
vae_patch_size = 1       # must match the trained VAE
latent_patch_size = 1    # DiT patch over the latent mesh dim (must divide L')
# latent_results_folder = 'results/airfoil_unsteady/latent_dit'
# # -----------------------------------

# if use_vae:
#     results_folder = latent_results_folder

# FA4 H200 compatibility: head_dim must be 32/64/128
if hidden_size % num_heads != 0:
    for h in [8, 6, 4, 3, 2, 16]:
        if hidden_size % h == 0 and (hidden_size // h) in (32, 64, 128):
            print(f"[FA4 Fix] num_heads {num_heads} -> {h} to match hidden {hidden_size}")
            num_heads = h
            break
else:
    hd = hidden_size // num_heads
    if hd not in (32, 64, 128):
        print(f"[FA4 Fix] head_dim {hd} not in (32,64,128), searching compatible heads")
        for h in [8, 6, 4, 3, 16, 2]:
            if hidden_size % h == 0 and (hidden_size // h) in (32, 64, 128):
                print(f"[FA4 Fix] num_heads {num_heads} -> {h} (head_dim {hidden_size//h})")
                num_heads = h
                break
        else:
            print(f"[FA4 Fix] No compatible heads for hidden {hidden_size}, switching to 256/4")
            hidden_size = 256
            num_heads = 4
if hidden_size == 192 and num_heads == 4 and _has_fa4:
    print("[Patch] Adjusting hidden 192->256 for FA4 head_dim 48->64 compatibility on H200")
    hidden_size = 256

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
# diffusion datasets below hold latents instead, so keep the fields aside.
fields_test_dataset = dataset_test

dit_patch_size = patch_size
model_shape = fields_shape  # (N, F, C, L) layout expected by VideoDiT
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

    # No TrainerVAE here: a second accelerator breaks the diffusion Trainer.
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

    def _to_dit_latents(z, conds, name):
        assert z.shape[0] == conds.shape[0], \
            f"{name}: {z.shape[0]} latents vs {conds.shape[0]} conditions — " \
            "the precomputed latents must come from the same splits/caps"
        return TensorDataset(z.permute(0, 2, 1, 3).contiguous(), conds)  # (N, F', C_lat, L')

    dataset_train = _to_dit_latents(z_train, dataset_train.tensors[1], "train")
    dataset_valid = _to_dit_latents(z_valid, dataset_valid.tensors[1], "valid")
    dataset_test = _to_dit_latents(z_test, fields_test_dataset.tensors[1], "test")

    dit_patch_size = latent_patch_size
    model_shape = dataset_train.tensors[0].shape
    print("Train latent shape (DiT layout):", tuple(model_shape), "(N, F', C_lat, L')")

model = VideoDiT(
    depth=depth,
    hidden_size=hidden_size,
    patch_size=dit_patch_size,
    num_frames=model_shape[1],
    num_heads=num_heads,
    input_size=model_shape[-1],
    cond_dim=dataset_train.tensors[1].shape[1],  # Mach, alpha
    class_dropout_prob=0.15,
    in_channels=model_shape[2],
    learn_sigma=False,
    use_swiglu=True,
    qk_norm=True,  # to avoid stability issues with bf16
    attn_type="vanilla",
    slice_num=128,
    mlp_ratio=2.5,
    factorize=factorize,
)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

flow_matching = create_flow_matching(
    neural_net=model,
    input_size=model_shape[-1],
    cond_scale=2.0,
    sampling_method="euler",
    num_sampling_steps=100,
    use_lognorm=use_lognorm,
)

trainer = Trainer(
    flow_matching,
    dataset=dataset_train,
    dataset_test=dataset_valid,  # validation only during training
    train_batch_size=1,
    train_lr=train_lr,
    train_num_steps=train_steps,
    gradient_accumulate_every=16,  # gradient accumulation steps
    ema_decay=ema_decay,  # exponential moving average decay
    amp=True,  # turn on mixed precision
    mixed_precision_type='bf16',
    results_folder=results_folder,  # folder to save results to
    save_and_sample_every=20000,
    eta_min_scheduler=1e-6,
    max_grad_norm=1.0,
    use_muon=True,
    muon_lr=muon_lr,
    muon_adjust_lr_fn=muon_adjust_lr_fn,
    muon_weight_decay=muon_weight_decay,
    # compile_model=True,
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
        "target_length": coefficients["target_length"],
        "original_length": coefficients["original_length"],
    },
    os.path.join(results_folder, "norm_stats.pt"),
)

trainer.train()
# trainer.load(5)

# Inference on the test set
trainer.ema.ema_model.eval()
# this will sample with multiple gpus, if available
if use_vae:
    latent_file = f"{results_folder}/latent_predictions.pt"
    if os.path.exists(latent_file):
        samples = torch.load(latent_file)
    else:
        samples, seqs = trainer.eval_model(dataset_test, batch_size=1, use_autocast=True)
else:
    if os.path.exists(f"{results_folder}/test_predictions_ema.pt"):
        samples = torch.load(f"{results_folder}/test_predictions_ema.pt")
    else:
        samples, seqs = trainer.eval_model(dataset_test, batch_size=1, use_autocast=True)

if trainer.accelerator.is_main_process:
    # ---- final evaluation (physical units, (N, F, C, L) fields) ----
    n_channels = channels
    channel_names = ["u", "v", "p", "rho"][:n_channels]
    orig_len = coefficients["original_length"]
    fmean = coefficients["fields_mean"].cpu()
    fstd = coefficients["fields_std"].cpu()
    gt_dataset = fields_test_dataset if use_vae else dataset_test
    gt_fields = gt_dataset.tensors[0].cpu()
    if use_vae:
        torch.save(samples, latent_file)
        # latents (N, F', C_lat, L') -> VAE layout -> denormalize -> decode
        z = samples.cpu().permute(0, 2, 1, 3).contiguous()
        z = denormalise_latents(z, os.path.join(vae_dir, "latents_stats.npz"))
        decoded = decode_latents(vae, z, batch_size=2, device=trainer.device)  # (N, C, F, L)
        decoded = decoded.permute(0, 2, 1, 3)[..., :orig_len]
        torch.save(decoded, f"{results_folder}/test_predictions_ema.pt")
        samples_denorm = decoded * fstd + fmean
        test_data_denorm = gt_fields[:, :keep_frames][..., :orig_len] * fstd + fmean
    else:
        torch.save(samples, f"{results_folder}/test_predictions_ema.pt")
        samples_denorm = samples.cpu()[..., :orig_len] * fstd + fmean
        test_data_denorm = gt_fields[..., :orig_len] * fstd + fmean
    conds_denorm = (gt_dataset.tensors[1].cpu() * coefficients["conds_std"].cpu()
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
