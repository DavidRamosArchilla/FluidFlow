"""Short DrivAerNet training run, to compare data parallel (DP) and hybrid data x
context parallel (DP x CP) training on the same global batches.

Trains the PointViT of train_drivaernet.py (same optimisation setup, shorter
cosine schedule) for --steps steps and saves to --out-dir/history.json the
per-step loss, the wall time and the validation MSE (EMA model, normalised
units, over --val-cars validation cars). With the same --seed, every --cp >= 1
sees the same cars at every step, so the loss curves should only differ by
floating point noise.

    torchrun --standalone --nproc-per-node=4 scripts/compare_parallelism_training.py --cp 4 --out-dir results/cp_compare/cp4
"""
import argparse
import json
import os
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from data.load_drivaernet import drivaernet_collate_fn, load_drivaernet
from fluidFlow.point_vit import PointViT
from fluidFlow.trainer import Trainer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cp", type=int, default=1, help="context parallel size (0 = plain DP path of the Trainer)")
    ap.add_argument("--backend", default="ulysses", choices=["ulysses", "allgather"])
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--global-batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--val-cars", type=int, default=50)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--cache-path", default="./data/driveaernet/drivaernet_cache.pt")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    dataset_train, dataset_valid, _, coefficients = load_drivaernet(args.cache_path)
    dataset_valid = Subset(dataset_valid, np.linspace(0, len(dataset_valid) - 1, args.val_cars).astype(int))

    model = PointViT(in_channels=3, out_channels=len(coefficients["fields_mean"]), patch_size=args.patch_size,
                     hidden_size=512, depth=12, num_heads=8, mlp_ratio=4, use_swiglu=True, qk_norm=True,
                     num_frequencies=6)
    trainer = Trainer(
        model, dataset_train, train_batch_size=args.global_batch, train_num_steps=args.steps, train_lr=2e-4,
        ema_decay=0.999, amp=True, mixed_precision_type="bf16", results_folder=args.out_dir,
        save_and_sample_every=10**9, eta_min_scheduler=4e-6, max_grad_norm=1.0, use_muon=True, muon_lr=2e-4,
        muon_adjust_lr_fn="match_rms_adamw", muon_weight_decay=1e-2, gradient_checkpointing=args.grad_ckpt,
        split_batches=True, dataloader_collate_fn=partial(drivaernet_collate_fn, pad_multiple=args.patch_size),
        save_eval_samples=False, compile_model=not args.no_compile,
        context_parallel_size=None if args.cp == 0 else args.cp, context_parallel_backend=args.backend)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    train_time = time.perf_counter() - t0
    peak_mem = torch.tensor(torch.cuda.max_memory_allocated() / 2**30, device="cuda")
    dist.all_reduce(peak_mem, op=dist.ReduceOp.MAX)

    # validation MSE of the EMA model (as at the Trainer milestones), whole cars on every GPU
    pred, true, mask = trainer.eval_model(dataset_valid, batch_size=4, use_autocast=True, return_mask=True)
    if trainer.accelerator.is_main_process:
        sq_err = (pred - true) ** 2
        val_mse = ((sq_err * mask.unsqueeze(1)).sum() / (mask.sum() * sq_err.shape[1])).item()
        result = {**vars(args), "world": trainer.accelerator.num_processes, "train_time_s": train_time,
                  "step_time_s": train_time / args.steps, "peak_mem_alloc_gb": peak_mem.item(),
                  "val_mse_ema": val_mse, "loss_history": trainer.loss_history}
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(result, f)
        print(f"RESULT cp={args.cp} backend={args.backend} train_time={train_time:.0f}s "
              f"step_time={train_time / args.steps:.3f}s final_loss={np.mean(trainer.loss_history[-100:]):.5f} "
              f"val_mse={val_mse:.5f}", flush=True)
    trainer.accelerator.wait_for_everyone()
    os._exit(0)  # skip the slow teardown of the persistent data loader workers


if __name__ == "__main__":
    main()
