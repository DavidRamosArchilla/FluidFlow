"""Cost of the ONERA CRM flow matching DiT (train_onera_crm.py) at small patch sizes,
with data parallelism (DP) or context parallelism (CP).

--mode train (torchrun): the Trainer for --warmup steps, then --steps timed steps;
appends to --out the median step time, model TFLOP/s per GPU and peak memory.
--mode sample (one process, one GPU): one classifier-free-guided denoising step of one
sample (what eval_model runs num_sampling_steps times, without CP).

The fields are random sequences of the padded ONERA length: the cost only depends on it.

    torchrun --standalone --nproc-per-node=4 scripts/benchmark_onera_cp.py --patch-size 2 --cp 4 --global-batch 1
    python scripts/benchmark_onera_cp.py --mode sample --patch-size 2
"""
import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from fluidFlow.dit import DiT
from fluidFlow.flow_matching import create_flow_matching
from fluidFlow.trainer import Trainer
from scripts.benchmark_parallelism import H100_BF16_PEAK_TFLOPS, model_flops_per_car

LENGTH = 260864  # the 260774 skin points padded as in train_onera_crm.py
CHANNELS, COND_DIM = 4, 3


class RandomOnera(torch.utils.data.Dataset):
    def __len__(self):
        return 312  # training cases

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return torch.randn(CHANNELS, LENGTH, generator=g), torch.randn(COND_DIM, generator=g)


def make_dit(patch_size):
    # the model of train_onera_crm.py
    return DiT(depth=12, hidden_size=512, patch_size=patch_size, num_heads=16, input_size=LENGTH, cond_dim=COND_DIM,
               class_dropout_prob=0.15, in_channels=CHANNELS, use_swiglu=True, attn_type="vanilla", qk_norm=True,
               mlp_ratio=4, use_rope=False)


def write(path, rec):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def bench_train(args):
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    rank, world = dist.get_rank(), dist.get_world_size()
    record = {"mode": "train", "patch_size": args.patch_size, "tokens": LENGTH // args.patch_size, "cp": args.cp,
              "backend": args.backend if args.cp > 1 else None, "world": world, "global_batch": args.global_batch,
              "samples_per_group": args.global_batch * args.cp // world, "grad_ckpt": args.grad_ckpt}
    torch.manual_seed(0)
    fm = create_flow_matching(neural_net=make_dit(args.patch_size), input_size=LENGTH, cond_scale=6.0,
                              sampling_method="euler", num_sampling_steps=500)
    try:
        trainer = Trainer(
            fm, RandomOnera(), train_batch_size=args.global_batch, train_lr=2e-4, train_num_steps=args.warmup,
            ema_decay=0.995, amp=True, mixed_precision_type="bf16", results_folder=tempfile.mkdtemp(prefix="onera_cp_"),
            save_and_sample_every=10**9, eta_min_scheduler=1e-6, max_grad_norm=1.0, compile_model=True,
            split_batches=True, use_muon=True, muon_lr=2e-4, muon_adjust_lr_fn="match_rms_adamw",
            muon_weight_decay=1e-2, gradient_checkpointing=args.grad_ckpt,
            context_parallel_size=args.cp, context_parallel_backend=args.backend)
        trainer.save_loss_plot = lambda: None
        # the Trainer waits for every GPU once at the end of each optimizer step: time stamp there
        stamps, barrier = [], trainer.accelerator.wait_for_everyone
        def timed_barrier():
            barrier()
            torch.cuda.synchronize()
            stamps.append(time.perf_counter())
        trainer.accelerator.wait_for_everyone = timed_barrier
        trainer.train()
        torch.cuda.reset_peak_memory_stats()
        trainer.train_num_steps += args.steps
        trainer.train()
    except Exception as e:
        oom = isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()
        if not oom:
            traceback.print_exc()
        write(args.out, {**record, "status": "OOM" if oom else "error", "rank": rank,
                         "error": traceback.format_exc().strip().splitlines()[-1]})
        os._exit(17)
    mem = torch.tensor(torch.cuda.max_memory_allocated() / 2**30, device="cuda")
    dist.all_reduce(mem, op=dist.ReduceOp.MAX)
    dt = np.diff(stamps[args.warmup - 1:])
    flops = model_flops_per_car([LENGTH], args.patch_size, 12, 512, 4).item() * args.global_batch
    step = float(np.median(dt))
    record.update({"status": "ok", "step_time_s": step, "step_times_s": dt.round(4).tolist(),
                   "samples_per_s": args.global_batch / step, "model_tflops_per_gpu": flops / step / world / 1e12,
                   "mfu": flops / step / world / 1e12 / H100_BF16_PEAK_TFLOPS, "peak_mem_alloc_gb": mem.item(),
                   "losses": trainer.loss_history[-3:]})
    if rank == 0:
        print("BENCH " + json.dumps(record), flush=True)
        write(args.out, record)
    dist.barrier()
    os._exit(0)


def bench_sample(args):
    record = {"mode": "sample", "patch_size": args.patch_size, "tokens": LENGTH // args.patch_size}
    torch.manual_seed(0)
    fm = create_flow_matching(neural_net=make_dit(args.patch_size), input_size=LENGTH).cuda().eval()
    net = torch.compile(fm.neural_net.forward_with_cond_scale)
    z, t, classes = torch.randn(1, CHANNELS, LENGTH, device="cuda"), torch.rand(1, device="cuda"), torch.randn(1, COND_DIM, device="cuda")
    times = []
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(args.warmup + args.steps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                net(z, t, classes, cond_scale=6.0)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
    except Exception as e:
        oom = isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()
        write(args.out, {**record, "status": "OOM" if oom else "error", "error": str(e).splitlines()[0][:200]})
        return
    step = float(np.median(times[args.warmup:]))
    record.update({"status": "ok", "denoising_step_s": step, "sample_500_steps_min": 500 * step / 60,
                   "peak_mem_alloc_gb": torch.cuda.max_memory_allocated() / 2**30})
    print("BENCH " + json.dumps(record), flush=True)
    write(args.out, record)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="train", choices=["train", "sample"])
    ap.add_argument("--patch-size", type=int, default=4)
    ap.add_argument("--cp", type=int, default=1, help="context parallel size (1 = data parallel)")
    ap.add_argument("--backend", default="ulysses", choices=["ulysses", "allgather"])
    ap.add_argument("--global-batch", type=int, default=4)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--out", default="results/onera_cp/bench.jsonl")
    args = ap.parse_args()
    bench_train(args) if args.mode == "train" else bench_sample(args)


if __name__ == "__main__":
    main()
