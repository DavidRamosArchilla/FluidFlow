"""Context parallelism must reproduce the single-GPU PointViT and DiT (also inside
FlowMatching): same predictions, loss and gradients, and the Trainer must take the
same optimizer steps for every (dp, cp) layout of the GPUs.

The sharding tests run anywhere with pytest. The multi-GPU tests need a GPU
node; they run under torchrun (no pytest needed) for every CP size dividing the
number of GPUs, both backends, and FA4 and SDPA attention:

    torchrun --nproc-per-node=4 tests/test_context_parallel.py [words: run only the checks named so]
"""
import gc
import os
import sys
import tempfile
import traceback
from datetime import timedelta
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for torchrun

import fluidFlow.attention as attention
from data.load_drivaernet import drivaernet_collate_fn
from fluidFlow.attention import Attention
from fluidFlow.context_parallel import BACKENDS, ContextParallel, balanced_token_starts, shard_batch, unshard
from fluidFlow.dit import DiT
from fluidFlow.point_vit import PointViT


def _cars(lengths, channels=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(channels, n, generator=g), torch.randn(3, n, generator=g)) for n in lengths]


# ---------------------------------------------------------------------------
# sharding (CPU, pytest)
# ---------------------------------------------------------------------------
def test_balanced_token_starts():
    for n in [0, 1, 5, 17, 1000, 1003]:
        for size in [1, 2, 3, 4, 8]:
            starts = balanced_token_starts(n, size)
            sizes = [b - a for a, b in zip(starts, starts[1:])]
            assert starts[0] == 0 and starts[-1] == n and max(sizes) - min(sizes) <= 1


def test_shard_unshard_roundtrip():
    lengths = [30005, 41000, 16, 37]
    batch = drivaernet_collate_fn(_cars(lengths), pad_multiple=16)
    for size in [1, 2, 3, 4]:
        shards = [shard_batch(batch, r, size, 16) for r in range(size)]
        for key in ("x", "context", "mask"):
            assert torch.equal(unshard([s[key] for s in shards], batch["mask"], 16), batch[key])
        # every point once, the same number of tokens per rank (+-1)
        assert sum(s["mask"].sum() for s in shards) == batch["mask"].sum()
        tokens = torch.stack([s["mask"].view(len(lengths), -1, 16).any(-1).sum(1) for s in shards])  # (size, B)
        assert (tokens.max(0).values - tokens.min(0).values <= 1).all()


def test_tuple_batch_sharding():
    """Tuple batches (fields, conditions) of equal-length sequences: the fields are split, the
    conditions copied, and the padding at the end of the last slices is masked out."""
    from fluidFlow.context_parallel import ShardCollate
    g = torch.Generator().manual_seed(0)
    samples = [(torch.randn(4, 4008, generator=g), torch.randn(3, generator=g)) for _ in range(2)]
    for size in [2, 3, 4]:
        shards = [ShardCollate(None, r, size, 4)(samples) for r in range(size)]
        full = torch.ones(2, 4008, dtype=torch.bool)
        x = unshard([s["x"] for s in shards], full, 4)
        assert torch.equal(x, torch.stack([f for f, _ in samples]))
        assert all(torch.equal(s["classes"], torch.stack([c for _, c in samples])) for s in shards)
        assert sum(s["mask"].sum() for s in shards) == 2 * 4008


# ---------------------------------------------------------------------------
# multi-GPU checks (torchrun)
# ---------------------------------------------------------------------------
def _flat_grads(module):
    return torch.cat([p.grad.flatten().float() for p in module.parameters() if p.grad is not None])


def _grad_metrics(grads, grads_ref):
    return {"grad_cosine": F.cosine_similarity(grads, grads_ref, dim=0).item(),
            "grad_norm_ratio": (grads.norm() / grads_ref.norm()).item()}


def _grads_ok(m):
    return m["grad_cosine"] > 0.999 and abs(m["grad_norm_ratio"] - 1) < 1e-2


def _shard_tokens(t, token_mask, cp):
    """(B, N, D) tokens -> this rank's (B, S, D) slice (shard_batch with patch_size 1)."""
    return shard_batch({"t": t.transpose(1, 2), "mask": token_mask}, cp.rank, cp.size, 1)["t"].transpose(1, 2)


def _gather_unshard(local, mask, cp, patch_size):
    """(B, ..., S * p) slices of the group -> (B, ..., L) unsharded."""
    chunks = [torch.empty_like(local) for _ in range(cp.size)]
    dist.all_gather(chunks, local.contiguous(), group=cp.group)
    return unshard(chunks, mask, patch_size)


def check_collective_adjoints(cp):
    """<A x, y> = <x, A^T y> summed over the GPUs, in float32, for the all-to-all and the
    all-gather: catches any permutation or scaling error of the backward exchanges."""
    from fluidFlow.context_parallel import _AllGather, _AllToAll
    torch.manual_seed(dist.get_rank())  # different data on every rank
    m = {}
    for name, op, shape in [("all_to_all", _AllToAll, (cp.size, 3, 5, 7)), ("all_gather", _AllGather, (3, 5, 7))]:
        x = torch.randn(*shape, device="cuda", dtype=torch.float64, requires_grad=True)
        out = op.apply(x, cp.group)
        y = torch.randn_like(out)
        (out * y).sum().backward()
        dots = torch.stack([(out * y).sum(), (x * x.grad).sum()]).detach()
        dist.all_reduce(dots)
        m[f"{name}_adjoint_rel_err"] = abs(dots[0].item() / dots[1].item() - 1)
    # broadcast: every GPU of the group ends with the draws of its first GPU
    for dtype_name, x in [("float", torch.rand(5, device="cuda")), ("bool", torch.rand(5, device="cuda") < 0.5)]:
        b = cp.broadcast(x).float()
        gathered = [torch.empty_like(b) for _ in range(cp.size)]
        dist.all_gather(gathered, b, group=cp.group)
        m[f"broadcast_{dtype_name}_max_diff"] = max((g - gathered[0]).abs().max().item() for g in gathered)
        if cp.rank == 0:
            m[f"broadcast_{dtype_name}_max_diff"] += (b - x.float()).abs().max().item()
    m["ok"] = all(v < 1e-6 for v in m.values())
    return m


def check_attention(cp, use_flash, B=3, N=1024, dim=512, heads=8):
    """One Attention layer on padded tokens: output, input and parameter gradients."""
    attention.is_flash_attn_available = use_flash
    torch.manual_seed(0)  # same weights and data on every rank
    module = Attention(dim, num_heads=heads, qkv_bias=True, qk_norm=True).cuda()
    x = torch.randn(B, N, dim, device="cuda")
    token_mask = torch.arange(N, device="cuda") < torch.tensor([N, 700, 333], device="cuda")[:, None]
    grad_out = torch.randn(B, N, dim, device="cuda") * token_mask.unsqueeze(-1)

    # reference: the whole sequence on this GPU
    xr = x.clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out_ref = module(xr, mask=token_mask)
    (out_ref.float() * grad_out).sum().backward()
    grads_ref = _flat_grads(module)
    module.zero_grad()

    # context parallel: this rank's slice of the tokens
    xl = _shard_tokens(x, token_mask, cp).requires_grad_()
    mask_l = shard_batch({"mask": token_mask}, cp.rank, cp.size, 1)["mask"]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = module(xl, cp=cp.prepare(mask_l))
    (out.float() * _shard_tokens(grad_out, token_mask, cp)).sum().backward()
    grads = _flat_grads(module)
    dist.all_reduce(grads, group=cp.group)  # the parameter gradient is the sum over the group

    real = mask_l
    m = {"out_max_err": (out.float() - _shard_tokens(out_ref.float(), token_mask, cp))[real].abs().max().item(),
         "x_grad_max_err": (xl.grad - _shard_tokens(xr.grad, token_mask, cp))[real].abs().max().item(),
         **_grad_metrics(grads, grads_ref)}
    m["ok"] = m["out_max_err"] < 3e-2 and m["x_grad_max_err"] < 3e-2 and _grads_ok(m)
    return m


def _point_vit(grad_ckpt):
    torch.manual_seed(0)
    model = PointViT(patch_size=16, hidden_size=512, depth=3, num_heads=8, num_frequencies=6).cuda()
    torch.nn.init.normal_(model.final_layer.weight, std=0.02)  # non-zero output (zero-initialised otherwise)
    if grad_ckpt:
        model.enable_gradient_checkpointing()
    return model


def check_point_vit(cp, use_flash, grad_ckpt=False, compile=False):
    """Whole PointViT on 3 cars of different lengths: predictions, loss, parameter gradients."""
    attention.is_flash_attn_available = use_flash
    model = _point_vit(grad_ckpt)
    batch = drivaernet_collate_fn(_cars([41000, 30005, 35555], seed=1), pad_multiple=16)
    full = {k: v.cuda() for k, v in batch.items()}
    local = {k: v.cuda() for k, v in shard_batch(batch, cp.rank, cp.size, 16).items()}

    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss_ref = model(full["x"], full["context"], full["mask"])
    loss_ref.backward()
    grads_ref = _flat_grads(model)
    model.zero_grad()

    if compile:
        torch._dynamo.reset()  # a fresh cache: the earlier checks compiled the same code
    fwd = torch.compile(model) if compile else model
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = fwd(local["x"], local["context"], local["mask"], cp=cp)
    loss.backward()
    # every CP group holds the same cars and the loss is normalised over all the GPUs,
    # so the sums over all the GPUs are the loss and gradient of these cars
    grads = _flat_grads(model)
    dist.all_reduce(grads)
    loss_group = cp.global_sum(loss.detach())

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_ref = model.predict(full["context"], full["mask"])
        pred = _gather_unshard(model.predict(local["context"], local["mask"], cp), full["mask"], cp, 16)
    m = {"loss_rel_err": abs(loss_group.item() / loss_ref.item() - 1),
         "pred_max_err": (pred - pred_ref).abs().max().item(),
         "pred_padding_zero": bool((pred[~full["mask"].unsqueeze(1).expand_as(pred)] == 0).all()),
         **_grad_metrics(grads, grads_ref)}
    m["ok"] = m["loss_rel_err"] < 1e-3 and m["pred_max_err"] < 3e-2 and m["pred_padding_zero"] and _grads_ok(m)
    return m


def check_trainer(world, backend, compile=False, steps=3, accumulate=1):
    """The Trainer takes the same optimizer steps (Muon + AdamW, bf16, clipping) for every
    (dp, cp) layout: the loss is the MSE of the global batch whatever the layout, and the
    global batches are the same cars (cars of different lengths, as in DrivAerNet)."""
    from fluidFlow.trainer import Trainer
    attention.is_flash_attn_available = attention_fa4
    lengths = torch.randint(16000, 32000, (16,), generator=torch.Generator().manual_seed(3)).tolist()
    dataset = [(f, c) for f, c in _cars(lengths, seed=2)]
    sizes = [s for s in (1, 2, 4, 8) if world % s == 0 and (backend != "ulysses" or 8 % s == 0)]
    results = {}
    for size in sizes:
        model = _point_vit(grad_ckpt=False)
        if compile:
            torch._dynamo.reset()
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(
                model, dataset, train_batch_size=8, train_num_steps=steps, train_lr=1e-3, muon_lr=1e-3,
                muon_adjust_lr_fn="match_rms_adamw", use_muon=True, max_grad_norm=1.0, amp=True,
                mixed_precision_type="bf16", save_and_sample_every=10**9, results_folder=tmp,
                dataloader_collate_fn=partial(drivaernet_collate_fn, pad_multiple=16), split_batches=True,
                gradient_accumulate_every=accumulate, compile_model=compile,
                context_parallel_size=size, context_parallel_backend=backend)
            trainer.train()
            params = torch.cat([p.detach().flatten().float() for p in trainer.accelerator.unwrap_model(trainer.model).parameters()])
            loss_history = torch.zeros(steps, device="cuda")
            if dist.get_rank() == 0:  # logged on the main process only
                loss_history.copy_(torch.tensor(trainer.loss_history))
            dist.broadcast(loss_history, src=0)
            results[size] = (params, loss_history)
            del trainer
            gc.collect()
    init = torch.cat([p.detach().flatten().float() for p in _point_vit(grad_ckpt=False).parameters()])
    params_dp, loss_dp = results[1]
    m = {}
    for size in sizes[1:]:
        params, loss = results[size]
        m[f"cp{size}_update_cosine"] = F.cosine_similarity(params - init, params_dp - init, dim=0).item()
        m[f"cp{size}_loss_max_rel_err"] = ((loss - loss_dp).abs() / loss_dp).max().item()
    # bf16 noise is ~1e-5; weighting the cars differently per layout would show up as ~1e-3
    m["ok"] = all(v > 0.999 for k, v in m.items() if k.endswith("cosine")) and \
        all(v < 3e-4 for k, v in m.items() if k.endswith("rel_err"))
    return m


# ---------------------------------------------------------------------------
# DiT and flow matching on tuple batches of equal-length sequences (e.g. ONERA CRM)
# ---------------------------------------------------------------------------
DIT_P, DIT_C = 4, 4
DIT_L = DIT_P * 1002  # 1002 tokens: not a multiple of cp, so the last slices end with padding


def _dit(rope=False, grad_ckpt=False, dropout=0.0):
    torch.manual_seed(0)
    model = DiT(input_size=DIT_L, patch_size=DIT_P, in_channels=DIT_C, cond_dim=3, hidden_size=512, depth=3,
                num_heads=8, use_swiglu=True, qk_norm=True, use_rope=rope, class_dropout_prob=dropout).cuda()
    # adaLN and the output layer start at zero: give them weights so that every path has gradients
    for name, p in model.named_parameters():
        if "adaLN_modulation" in name or "final_layer.linear" in name:
            torch.nn.init.normal_(p, std=0.02)
    if grad_ckpt:
        model.enable_gradient_checkpointing()
    return model


def _dit_batch(B=2, seed=4):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, DIT_C, DIT_L, generator=g).cuda(), torch.randn(B, 3, generator=g).cuda()


def _shard_sequences(t, cp):
    """(B, ..., L) equal-length sequences -> this rank's slice and its (B, L_local) point mask."""
    full = torch.ones(t.shape[0], t.shape[-1], dtype=torch.bool, device=t.device)
    local = shard_batch({"t": t, "mask": full}, cp.rank, cp.size, DIT_P)
    return local["t"], local["mask"]


def check_dit(cp, use_flash, rope=False, grad_ckpt=False, compile=False):
    """DiT on sequences split over the CP group (sin-cos positions, RoPE, adaLN conditioning,
    padded slices): output, input gradient and parameter gradients."""
    attention.is_flash_attn_available = use_flash
    model = _dit(rope, grad_ckpt)
    x, classes = _dit_batch()
    t = torch.tensor([0.3, 0.8], device="cuda")
    drop = torch.tensor([False, True], device="cuda")  # a conditional and an unconditional sample
    grad_out = torch.randn_like(x)

    xr = x.clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out_ref = model(xr, t, classes, force_drop_ids=drop)
    (out_ref.float() * grad_out).sum().backward()
    grads_ref = _flat_grads(model)
    model.zero_grad()

    xl, mask = _shard_sequences(x, cp)
    xl.requires_grad_()
    if compile:
        torch._dynamo.reset()
    fwd = torch.compile(model) if compile else model
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = fwd(xl, t, classes, force_drop_ids=drop, mask=mask, cp=cp)
    (out.float() * _shard_sequences(grad_out, cp)[0] * mask.unsqueeze(1)).sum().backward()
    grads = _flat_grads(model)
    dist.all_reduce(grads, group=cp.group)

    full = torch.ones(x.shape[0], DIT_L, dtype=torch.bool, device="cuda")
    pred = _gather_unshard(out.detach().float(), full, cp, DIT_P)
    x_grad = _gather_unshard(xl.grad, full, cp, DIT_P)
    m = {"out_rel_err": ((pred - out_ref.float()).abs().max() / out_ref.float().abs().max()).item(),
         "x_grad_rel_err": ((x_grad - xr.grad).abs().max() / xr.grad.abs().max()).item(),
         **_grad_metrics(grads, grads_ref)}
    m["ok"] = m["out_rel_err"] < 3e-2 and m["x_grad_rel_err"] < 3e-2 and _grads_ok(m)
    return m


def check_flow_matching(cp):
    """FlowMatching(DiT) loss and gradients on sequences split over the CP group. The noise is
    the one of the single-GPU reference; the diffusion times differ on every GPU (as with the
    Trainer's per-GPU seeds), so they only match if the group takes the time of its first GPU."""
    from fluidFlow.flow_matching import create_flow_matching
    attention.is_flash_attn_available = attention_fa4
    fm = create_flow_matching(neural_net=_dit(), input_size=DIT_L).cuda()
    x, classes = _dit_batch()
    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(5)).cuda()
    t_ref = torch.tensor([0.3, 0.8], device="cuda")
    transport = fm.sampler.transport

    def sample(x1, *args):
        if x1.shape[-1] == DIT_L:
            return t_ref, noise, x1
        return t_ref + 0.1 * cp.rank, _shard_sequences(noise, cp)[0], x1
    transport.sample = sample

    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss_ref = fm(x, classes)
    loss_ref.backward()
    grads_ref = _flat_grads(fm)
    fm.zero_grad()
    xl, mask = _shard_sequences(x, cp)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = fm(xl, classes, mask=mask, cp=cp)
    loss.backward()
    del transport.sample
    # every CP group holds the same samples and the loss is normalised over all the GPUs
    grads = _flat_grads(fm)
    dist.all_reduce(grads)
    m = {"loss_rel_err": abs(cp.global_sum(loss.detach()).item() / loss_ref.item() - 1), **_grad_metrics(grads, grads_ref)}
    m["ok"] = m["loss_rel_err"] < 1e-3 and _grads_ok(m)
    return m


def check_flow_matching_trainer(world, steps=3):
    """The Trainer with FlowMatching(DiT) on a TensorDataset of (fields, conditions), as
    scripts/train_onera_crm.py, for every (dp, cp) layout: the same parameters on all the GPUs
    after the steps, and a first loss independent of the layout (the noise differs, so ~1%)."""
    from torch.utils.data import TensorDataset
    from fluidFlow.flow_matching import create_flow_matching
    from fluidFlow.trainer import Trainer
    attention.is_flash_attn_available = attention_fa4
    g = torch.Generator().manual_seed(6)
    dataset = TensorDataset(torch.randn(16, DIT_C, DIT_L, generator=g), torch.randn(16, 3, generator=g))
    m, first_loss = {}, {}
    for size in [s for s in (1, 2, 4, 8) if world % s == 0]:
        fm = create_flow_matching(neural_net=_dit(dropout=0.2), input_size=DIT_L)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(
                fm, dataset, train_batch_size=8, train_num_steps=steps, train_lr=1e-3, muon_lr=1e-3,
                muon_adjust_lr_fn="match_rms_adamw", use_muon=True, max_grad_norm=1.0, amp=True,
                mixed_precision_type="bf16", save_and_sample_every=10**9, results_folder=tmp, split_batches=True,
                context_parallel_size=size, context_parallel_backend="ulysses")
            trainer.train()
            params = torch.cat([p.detach().flatten().float() for p in trainer.accelerator.unwrap_model(trainer.model).parameters()])
            params_rank0 = params.clone()
            dist.broadcast(params_rank0, src=0)
            loss = torch.zeros(steps, device="cuda")
            if dist.get_rank() == 0:  # logged on the main process only
                loss.copy_(torch.tensor(trainer.loss_history))
            dist.broadcast(loss, src=0)
            m[f"cp{size}_param_diff_between_gpus"] = (params - params_rank0).abs().max().item()
            first_loss[size] = loss[0].item()
            m[f"cp{size}_losses_finite"] = bool(torch.isfinite(loss).all())
            del trainer
            gc.collect()
    for size, value in first_loss.items():
        m[f"cp{size}_first_loss"] = value
    m["ok"] = all(v == 0 for k, v in m.items() if k.endswith("between_gpus")) and \
        all(v for k, v in m.items() if k.endswith("finite")) and \
        max(first_loss.values()) / min(first_loss.values()) < 1.1
    return m


attention_fa4 = attention.is_flash_attn_available


def main():
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    world, rank = dist.get_world_size(), dist.get_rank()
    flash_options = [True, False] if attention_fa4 else [False]
    only = sys.argv[1:]  # run only the checks whose name contains one of these words
    results = []

    def run(name, fn, *args, **kwargs):
        if only and not any(word in name for word in only):
            return
        try:
            m = fn(*args, **kwargs)
        except Exception:
            traceback.print_exc()
            m = {"ok": False, "error": traceback.format_exc().strip().splitlines()[-1]}
        ok = torch.tensor(float(m["ok"]), device="cuda")
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)  # passes only if it passes on every rank
        m["ok"] = bool(ok.item())
        results.append((name, m))
        if rank == 0:
            print(f"[{'PASS' if m['ok'] else 'FAIL'}] {name}: " +
                  ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items() if k != "ok"),
                  flush=True)

    for size in [s for s in (2, 4, 8) if s <= world and world % s == 0]:
        run(f"collective adjoints cp={size}", check_collective_adjoints, ContextParallel(size))
        for backend in BACKENDS:
            if backend == "ulysses" and 8 % size:
                continue  # the test models have 8 heads
            cp = ContextParallel(size, backend)
            for use_flash in flash_options:
                tag = f"cp={size} {backend} {'fa4' if use_flash else 'sdpa'}"
                run(f"attention {tag}", check_attention, cp, use_flash)
                run(f"point_vit {tag}", check_point_vit, cp, use_flash)
            run(f"point_vit cp={size} {backend} grad checkpointing", check_point_vit, cp, attention_fa4, grad_ckpt=True)
            run(f"point_vit cp={size} {backend} torch.compile", check_point_vit, cp, attention_fa4, compile=True)
            for use_flash in flash_options:
                run(f"dit cp={size} {backend} {'fa4' if use_flash else 'sdpa'}", check_dit, cp, use_flash)
            run(f"dit cp={size} {backend} rope", check_dit, cp, attention_fa4, rope=True)
            run(f"dit cp={size} {backend} rope, grad checkpointing", check_dit, cp, attention_fa4, rope=True,
                grad_ckpt=True)
            run(f"dit cp={size} {backend} torch.compile", check_dit, cp, attention_fa4, compile=True)
            run(f"flow matching dit cp={size} {backend}", check_flow_matching, cp)
    for backend in BACKENDS:
        run(f"trainer steps dp x cp layouts {backend}", check_trainer, world, backend)
    run("trainer steps dp x cp layouts ulysses, gradient accumulation 2", check_trainer, world, "ulysses",
        accumulate=2)
    run("trainer steps dp x cp layouts ulysses torch.compile", check_trainer, world, "ulysses", compile=True)
    run("trainer flow matching dit, tuple batches, dp x cp layouts", check_flow_matching_trainer, world)

    failed = [name for name, m in results if not m["ok"]]
    if rank == 0:
        print(f"\n{len(results) - len(failed)}/{len(results)} checks passed" +
              ("" if not failed else "; failed: " + "; ".join(failed)), flush=True)
    dist.destroy_process_group()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
