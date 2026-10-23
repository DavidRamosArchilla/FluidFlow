"""Throughput and memory of data parallel (DP) vs hybrid data x context parallel
(DP x CP) training of the DrivAerNet PointViT (the model of train_drivaernet.py).

Runs the real Trainer for --warmup steps (torch.compile, FA4 JIT, data loader
start-up), times --steps more and appends one JSON line to --out with the step
time, cars/s, tokens/s and model TFLOP/s per GPU and the peak GPU memory (max
over the GPUs). A run that does not fit in memory writes "status": "OOM".

    torchrun --standalone --nproc-per-node=4 scripts/benchmark_parallelism.py --patch-size 16 --cp 4
    srun --ntasks-per-node=1 bash scripts/slurm/torchrun_nodes.sh scripts/benchmark_parallelism.py --cp 4 ...

--cp 0 is the plain DP path of the Trainer (context_parallel_size=None); --cp 1
is DP through the hybrid data path, which draws the same global batches as
every --cp > 1 for the same seed (use it for the DP vs DP x CP comparisons).
"""
import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from datetime import timedelta
from functools import partial

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from data.load_drivaernet import drivaernet_collate_fn, load_drivaernet
from fluidFlow.point_vit import PointViT
from fluidFlow.trainer import Trainer

H100_BF16_PEAK_TFLOPS = 989.0  # dense, SXM


class SyntheticCars(torch.utils.data.Dataset):
    """Random cars with given point counts (no cache needed; the cost only depends on the lengths)."""
    def __init__(self, lengths, channels=1, seed=0):
        self.lengths = [int(n) for n in lengths]
        self.channels = channels
        self.seed = seed

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(self.seed + i)
        n = self.lengths[i]
        return torch.randn(self.channels, n, generator=g), torch.randn(3, n, generator=g)


def model_flops_per_car(num_points, patch_size, depth, hidden, mlp_ratio):
    """Training FLOPs of one car (3x the forward: forward + backward, no recomputation).

    Per token and layer the forward does 2 * (4 D^2 + 3 D F) for qkv, proj and
    the SwiGLU MLP (F = 2/3 * mlp_ratio * D) and 4 N D for attention (QK^T and PV
    over the N tokens of the car); patch embedding / output layer are negligible.
    """
    n = np.ceil(np.asarray(num_points, dtype=np.float64) / patch_size)  # tokens
    f = int(2 / 3 * hidden * mlp_ratio)
    dense = 2 * (4 * hidden**2 + 3 * hidden * f) * n
    attn = 4 * n**2 * hidden
    return 3 * depth * (dense + attn)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--mlp-ratio", type=float, default=4)
    ap.add_argument("--num-frequencies", type=int, default=6)
    ap.add_argument("--cp", type=int, default=1, help="context parallel size (0 = plain DP path of the Trainer)")
    ap.add_argument("--backend", default="ulysses", choices=["ulysses", "allgather"])
    ap.add_argument("--global-batch", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="random cars with the real point counts")
    ap.add_argument("--cache-path", default="./data/driveaernet/drivaernet_cache.pt")
    ap.add_argument("--out", default="results/cp_bench/bench.jsonl")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    # our own process group with a short timeout, so that a rank that dies (e.g. OOM)
    # does not leave the others waiting for 30 min; the Trainer's Accelerator reuses it
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank, world = dist.get_rank(), dist.get_world_size()

    torch.manual_seed(0)
    np.random.seed(0)
    if args.synthetic:
        lengths = np.random.default_rng(0).integers(359432, 553448, size=5817)
        dataset = SyntheticCars(lengths)
    else:
        dataset, _, _, _ = load_drivaernet(args.cache_path)
        lengths = dataset.offsets.diff().numpy()

    cp = None if args.cp == 0 else args.cp
    record = {
        "tag": args.tag, "patch_size": args.patch_size, "depth": args.depth, "hidden": args.hidden,
        "heads": args.heads, "cp": args.cp, "backend": args.backend if (cp or 0) > 1 else None,
        "world": world, "nodes": int(os.environ.get("SLURM_STEP_NUM_NODES", os.environ.get("SLURM_NNODES", 1))),
        "dp": world // (cp or 1), "global_batch": args.global_batch,
        "cars_per_group": args.global_batch // (world // (cp or 1)),
        "grad_ckpt": args.grad_ckpt, "compile": not args.no_compile, "synthetic": args.synthetic,
        "mean_tokens_per_car": float(np.ceil(lengths / args.patch_size).mean()),
    }

    def write(rec):
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")

    model = PointViT(in_channels=3, out_channels=1, patch_size=args.patch_size, hidden_size=args.hidden,
                     depth=args.depth, num_heads=args.heads, mlp_ratio=args.mlp_ratio, use_swiglu=True,
                     qk_norm=True, num_frequencies=args.num_frequencies)
    tmp = tempfile.mkdtemp(prefix="cp_bench_")
    try:
        # the optimisation setup of train_drivaernet.py
        trainer = Trainer(
            model, dataset, train_batch_size=args.global_batch, train_num_steps=args.warmup, train_lr=2e-4,
            ema_decay=0.999, amp=True, mixed_precision_type="bf16", results_folder=tmp,
            save_and_sample_every=10**9, eta_min_scheduler=4e-6, max_grad_norm=1.0, use_muon=True, muon_lr=2e-4,
            muon_adjust_lr_fn="match_rms_adamw", muon_weight_decay=1e-2, gradient_checkpointing=args.grad_ckpt,
            split_batches=True, dataloader_collate_fn=partial(drivaernet_collate_fn, pad_multiple=args.patch_size),
            save_eval_samples=False, compile_model=not args.no_compile,
            context_parallel_size=cp, context_parallel_backend=args.backend)
        trainer.save_loss_plot = lambda: None  # not part of a training step
        # the Trainer waits for every GPU once at the end of each optimizer step: time stamp there
        stamps, barrier = [], trainer.accelerator.wait_for_everyone
        def timed_barrier():
            barrier()
            torch.cuda.synchronize()
            stamps.append(time.perf_counter())
        trainer.accelerator.wait_for_everyone = timed_barrier
        trainer.train()  # warm-up (compilation, FA4 JIT, recompilation for the dynamic lengths)
        torch.cuda.reset_peak_memory_stats()
        trainer.train_num_steps += args.steps
        trainer.train()
    except Exception as e:
        oom = isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()
        if not oom:
            traceback.print_exc()
        print(f"[rank {rank}] {'OOM' if oom else 'error'}", flush=True)
        write({**record, "status": "OOM" if oom else "error", "rank": rank,
               "error": traceback.format_exc().strip().splitlines()[-1]})
        os._exit(17)  # the launcher tears the other ranks down

    mem = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                       dtype=torch.float64, device="cuda")
    dist.all_reduce(mem, op=dist.ReduceOp.MAX)
    # per-step times of the timed steps, and the model FLOPs of the cars of each step: with the
    # hybrid data path, global batch k is perm[k B:(k + 1) B] of the DistributedSampler permutation
    dt = np.diff(stamps[args.warmup - 1:])
    steps = np.arange(args.warmup, args.warmup + args.steps)
    flops_car = model_flops_per_car(lengths, args.patch_size, args.depth, args.hidden, args.mlp_ratio)
    tokens_car = np.ceil(lengths / args.patch_size)
    if cp is None:  # accelerate's sampler: the mean car
        flops_step = np.full(args.steps, flops_car.mean() * args.global_batch)
        tokens_step = np.full(args.steps, tokens_car.mean() * args.global_batch)
    else:
        perm = torch.randperm(len(lengths), generator=torch.Generator().manual_seed(0)).numpy()  # epoch 0
        cars = [perm[k * args.global_batch:(k + 1) * args.global_batch] for k in steps]
        flops_step = np.array([flops_car[c].sum() for c in cars])
        tokens_step = np.array([tokens_car[c].sum() for c in cars])
    step_time = float(np.median(dt))
    tflops_per_gpu = float(np.median(flops_step / dt)) / world / 1e12
    record.update({
        "status": "ok", "step_time_s": step_time, "step_time_mean_s": float(dt.mean()),
        "step_times_s": dt.round(4).tolist(), "cars_per_s": args.global_batch / step_time,
        "tokens_per_s_per_gpu": float(np.median(tokens_step / dt)) / world,
        "model_tflops_per_gpu": tflops_per_gpu, "mfu": tflops_per_gpu / H100_BF16_PEAK_TFLOPS,
        "flops_from_actual_batches": cp is not None,
        "peak_mem_alloc_gb": mem[0].item() / 2**30, "peak_mem_reserved_gb": mem[1].item() / 2**30,
        "final_loss": trainer.loss_history[-1] if trainer.loss_history else None,
    })
    if rank == 0:
        print("BENCH " + json.dumps(record), flush=True)
        write(record)
    dist.barrier()
    os._exit(0)  # skip the slow teardown of the persistent data loader workers


if __name__ == "__main__":
    main()
