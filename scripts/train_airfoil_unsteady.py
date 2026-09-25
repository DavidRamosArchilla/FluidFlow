from data.load_airfoil_unsteady import load_airfoil_unsteady
from fluidFlow.video_dit import VideoDiT
from fluidFlow.trainer import Trainer
from fluidFlow.flow_matching import create_flow_matching

import os
import shutil

import numpy as np
import torch


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
channels = 3              # 3->[u,v,p]  4->[u,v,p,rho]
time_frames = 601         # temporal subsampling (None for the full 601 frames)
time_mode = "uniform"     # "uniform" or "first"
max_train_samples = None
max_valid_samples = None
max_test_samples = None
# ---------- model config ----------
patch_size = 8
depth = 8
hidden_size = 384
num_heads = 12
factorize = True          # True if time_frames > 32 to save memory
# ---------- training config ----------
train_lr = 1e-4
train_steps = 100000
results_folder = 'results/airfoil_unsteady_experiment'
# -----------------------------------

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

dataset_train, dataset_valid, dataset_test, coefficients = load_airfoil_unsteady(
    data_dir,
    channels=channels,
    time_frames=time_frames,
    time_mode=time_mode,
    max_train_samples=max_train_samples,
    max_valid_samples=max_valid_samples,
    max_test_samples=max_test_samples,
    patch_size=patch_size,
)

fields_shape = dataset_train.tensors[0].shape  # (N, F, C, Nnodes)
print("Train fields shape", fields_shape)

model = VideoDiT(
    depth=depth,
    hidden_size=hidden_size,
    patch_size=patch_size,
    num_frames=fields_shape[1],
    num_heads=num_heads,
    input_size=fields_shape[-1],
    cond_dim=dataset_train.tensors[1].shape[1],  # Mach, alpha
    class_dropout_prob=0.15,
    in_channels=fields_shape[2],
    learn_sigma=False,
    use_swiglu=True,
    qk_norm=True,  # to avoid stability issues with bf16
    attn_type="physics",
    slice_num=128,
    mlp_ratio=2.5,
    factorize=factorize,
)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

flow_matching = create_flow_matching(
    neural_net=model,
    input_size=fields_shape[-1],
    cond_scale=2.0,
    sampling_method="euler",
    num_sampling_steps=100,
)

trainer = Trainer(
    flow_matching,
    dataset=dataset_train,
    dataset_test=dataset_valid,  # validation only during training
    train_batch_size=1,
    train_lr=train_lr,
    train_num_steps=train_steps,
    gradient_accumulate_every=16,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=True,  # turn on mixed precision
    mixed_precision_type='bf16',
    results_folder=results_folder,  # folder to save results to
    save_and_sample_every=20000,
    eta_min_scheduler=1e-6,
    max_grad_norm=1.0,
    use_muon=True,
    compile_model=True,
    split_batches=True,
)

shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))

trainer.train()

# Inference on the test set (no metric evaluation for the moment)
trainer.ema.ema_model.eval()
# this will sample with multiple gpus, if available
samples, seqs = trainer.eval_model(dataset_test, batch_size=1, use_autocast=True)

if trainer.accelerator.is_main_process:
    torch.save(samples, f"{results_folder}/test_predictions_ema.pt")
