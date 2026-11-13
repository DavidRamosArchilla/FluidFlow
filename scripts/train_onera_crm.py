from data.load_onera_crm import load_onera_crm, NWALLP
from fluidFlow.dit import DiT
from fluidFlow.trainer import Trainer
from fluidFlow.flow_matching import create_flow_matching
from fluidFlow.evaluation import RegressionEvaluator

import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset


# you need first to download the data --> https://entrepot.recherche.data.gouv.fr/file.xhtml?persistentId=doi:10.57745/Z9LDY8&version=2.0
data_dir = "../denoising-diffusion-pytorch-fluid-mechanics-main/data/onera_data"

# ---------- model / parallelism ----------
patch_size = 1                # points per token (1: every skin point is a token, 260864 tokens)
# GPUs per sample (None = data parallel); ulysses needs num_heads % cp == 0. With one sample
# per CP group the job uses train_batch_size * cp GPUs: 8 -> 256 GPUs (2.1 s/step, each sample
# on 2 nodes), 4 -> 128 GPUs (4.05 s/step, a sample inside a node)
context_parallel_size = 8
# ---------- training ----------
train_batch_size = 32         # global batch: one sample per CP group
train_steps = 50000           # 1.2M samples, as the 300k steps x 4 of the p = 4 recipe (~22 h on 256 GPUs)
save_and_sample_every = 5000  # checkpoint + validation (10 milestones)
# ---------- evaluation ----------
val_cases = 32                # test cases sampled at every milestone (the final evaluation uses all)
val_sampling_steps = 100      # ODE steps of that validation (the final evaluation uses num_sampling_steps)
num_sampling_steps = 500

# pad the 260774 skin points to a whole number of patches (and of patches per GPU with
# context parallelism): 260864 = 256 x 1019
dataset_train, dataset_test, coefficients = load_onera_crm(data_dir, pad_length=260864)
dataset_val = Subset(dataset_test, np.linspace(0, len(dataset_test) - 1, val_cases).astype(int).tolist())

model = DiT(
    depth=12,
    hidden_size=512,
    patch_size=patch_size,
    num_heads=16,
    input_size=dataset_train.tensors[0].shape[2],
    cond_dim=dataset_train.tensors[1].shape[1],
    class_dropout_prob=0.15,
    in_channels=dataset_train.tensors[0].shape[1],
    use_swiglu=True,
    attn_type="vanilla",
    qk_norm=True, # to avoid stability issues with bf16
    mlp_ratio=4,
    use_rope=False
)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

flow_matching = create_flow_matching(
    neural_net=model,
    input_size=dataset_train.tensors[0].shape[2],
    cond_scale=6.0,
    sampling_method="euler",
    num_sampling_steps=num_sampling_steps,
)

results_folder = f'results/onera_crm/p{patch_size}_d512_vanilla_b{train_batch_size}'
trainer = Trainer(
    flow_matching,
    dataset=dataset_train,
    dataset_test=dataset_val,  # validation MSE (normalised units) at every milestone
    train_batch_size=train_batch_size,
    train_lr=2e-4,
    train_num_steps=train_steps,  # total training steps
    gradient_accumulate_every=1,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp = True,                       # turn on mixed precision
    mixed_precision_type='bf16',
    results_folder=results_folder,  # folder to save results to
    save_and_sample_every=save_and_sample_every,
    eval_sampling_kwargs={"num_steps": val_sampling_steps},
    eta_min_scheduler=1e-6,
    max_grad_norm=1.0,
    compile_model=True,
    split_batches=True,
    context_parallel_size=context_parallel_size,
    context_parallel_backend="ulysses",
    use_muon=True,
    muon_lr=2e-4,
    muon_adjust_lr_fn="match_rms_adamw",
    muon_weight_decay=1e-2,
)

if trainer.accelerator.is_main_process:
    shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
    torch.save(coefficients, os.path.join(results_folder, "norm_stats.pt"))

# resume from the last checkpoint if the job was interrupted and submitted again
milestones = sorted(int(p.stem.split("-")[1]) for p in Path(results_folder).glob("model-*.pt"))
if milestones:
    trainer.accelerator.print(f"Resuming from {results_folder}/model-{milestones[-1]}.pt")
    trainer.load(milestones[-1])

trainer.train()

# Evaluate the model on the whole test set and save predictions
trainer.ema.ema_model.eval()
# this will sample with multiple gpus, if available (one test case per GPU and batch)
samples, seqs = trainer.eval_model(dataset_test, batch_size=trainer.accelerator.num_processes, use_autocast=True)

if trainer.accelerator.is_main_process:
    # drop the padding
    samples = (samples[..., :NWALLP].cpu() * coefficients['train_std']) + coefficients['train_mean']
    torch.save(samples, f"{results_folder}/test_predictions_ema.pt")
    test_data = (dataset_test.tensors[0][..., :NWALLP].cpu() * coefficients['train_std']) + coefficients['train_mean']
    evaluator = RegressionEvaluator()
    metrics = evaluator(test_data, samples)
    evaluator.print_metrics()
    with open(f"{results_folder}/metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
