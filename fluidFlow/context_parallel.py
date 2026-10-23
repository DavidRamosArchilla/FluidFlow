"""Context parallelism (CP) combined with data parallelism (DP).

The GPUs form a (dp, cp) grid. The ``cp`` GPUs of a group work on the same
samples, each one holding a contiguous slice of the tokens of every sample, and
the ``dp`` groups see different samples (the global batch is split over the
groups). Only attention mixes tokens, so the patch embedding, MLPs, norms,
output layer and loss run on the local tokens unchanged, and attention
exchanges what it needs over the CP group:

- ``"ulysses"`` (DeepSpeed-Ulysses): an all-to-all turns (local tokens, all
  heads) into (all tokens, num_heads / cp heads), every GPU runs exact
  attention for its heads with the same FA4 kernel as without CP, and a second
  all-to-all turns the output back. Needs ``num_heads % cp == 0``.
- ``"allgather"`` (Llama 3 style): K and V are all-gathered along the sequence
  and the local queries attend to all the keys. Works for any cp, but the
  gathered K, V of every layer are kept for the backward.

Gradients flow through the collectives (an all-to-all is its own transpose, the
transpose of an all-gather is a reduce-scatter), so after ``backward`` every
GPU holds the gradient of the summed loss of its group with respect to its copy
of the parameters. Each GPU's loss is its share of the MSE of the global batch
(its squared errors over the real points of all the GPUs), so the objective
does not depend on the (dp, cp) layout; :class:`fluidFlow.trainer.Trainer`
multiplies it by the number of GPUs because DDP averages the gradients instead
of summing them.

The gathered sequence is in rank order, not in the original token order: this
is exact for models without positional encodings inside attention, such as
:class:`fluidFlow.point_vit.PointViT` (positions enter through the point
coordinates).

Usage (the Trainer does this when ``context_parallel_size`` is set)::

    cp = ContextParallel(size=4, backend="ulysses")    # after init_process_group
    batch = shard_batch(collate_fn(samples), cp.rank, cp.size, patch_size)
    loss = model(batch["x"], batch["context"], batch["mask"], cp=cp)
"""

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

BACKENDS = ("ulysses", "allgather")

# renamed in recent PyTorch (same signatures)
_all_gather_single = getattr(dist, "all_gather_single", None) or dist.all_gather_into_tensor
_reduce_scatter_single = getattr(dist, "reduce_scatter_single", None) or dist.reduce_scatter_tensor


# ---------------------------------------------------------------------------
# sharding of the padded batches (runs in the DataLoader workers)
# ---------------------------------------------------------------------------
def balanced_token_starts(num_tokens, size):
    """Split ``num_tokens`` into ``size`` contiguous ranges whose lengths differ by
    at most one. Rank r gets the tokens ``[starts[r], starts[r + 1])``."""
    q, r = divmod(num_tokens, size)
    return [i * q + min(i, r) for i in range(size + 1)]


def shard_batch(batch, rank, size, patch_size):
    """This rank's slice of every sample of a padded dict batch.

    ``batch["mask"]`` (B, L) marks the real points, which come first in every
    row (as built by ``drivaernet_collate_fn``); L is a multiple of
    ``patch_size``. The n_i points of sample i form ceil(n_i / patch_size)
    tokens, split over the ranks in contiguous ranges that differ by at most
    one token, so every rank gets the same amount of work. Every tensor whose
    first dim is B and last dim is L is sliced and zero-padded to
    ``ceil(max_i tokens_i / size)`` local tokens; other entries are kept.
    """
    mask = batch["mask"]
    B, L = mask.shape
    p = patch_size
    assert L % p == 0, f"padded length {L} must be a multiple of patch_size {p}"
    num_tokens = [math.ceil(n / p) for n in mask.sum(dim=1).tolist()]
    local_tokens = max(1, math.ceil(max(num_tokens) / size))
    out = {}
    for key, t in batch.items():
        if not (torch.is_tensor(t) and t.dim() >= 2 and t.shape[0] == B and t.shape[-1] == L):
            out[key] = t
            continue
        local = t.new_zeros(*t.shape[:-1], local_tokens * p)
        for i, n in enumerate(num_tokens):
            starts = balanced_token_starts(n, size)
            a, b = starts[rank] * p, starts[rank + 1] * p
            local[i, ..., :b - a] = t[i, ..., a:b]
        out[key] = local
    return out


def unshard(chunks, mask, patch_size):
    """Inverse of :func:`shard_batch` for one tensor.

    chunks: the (B, ..., S * patch_size) slices of every rank, in rank order.
    mask: the (B, L) point mask of the unsharded batch.
    Returns the (B, ..., L) tensor with every sample's points back in order.
    """
    size, p = len(chunks), patch_size
    B, L = mask.shape
    out = chunks[0].new_zeros(*chunks[0].shape[:-1], L)
    for i, n in enumerate(mask.sum(dim=1).tolist()):
        starts = balanced_token_starts(math.ceil(n / p), size)
        for r, chunk in enumerate(chunks):
            a, b = starts[r] * p, starts[r + 1] * p
            out[i, ..., a:b] = chunk[i, ..., :b - a]
    return out


class ShardCollate:
    """collate_fn wrapper keeping the slice of one CP rank (picklable, for workers)."""
    def __init__(self, collate_fn, rank, size, patch_size):
        self.collate_fn = collate_fn
        self.rank = rank
        self.size = size
        self.patch_size = patch_size

    def __call__(self, samples):
        batch = self.collate_fn(samples)
        if not (isinstance(batch, dict) and "mask" in batch):
            raise TypeError("context parallelism needs dict batches with a (B, L) 'mask' (see drivaernet_collate_fn)")
        # every rank must get tokens of every sample (empty varlen segments are untested in FA4)
        min_points = int(batch["mask"].sum(dim=1).min())
        if min_points <= (self.size - 1) * self.patch_size:
            raise ValueError(f"a sample with {min_points} points has fewer tokens than the {self.size} CP ranks")
        return shard_batch(batch, self.rank, self.size, self.patch_size)


def cycle_epochs(dl):
    """Endless iterator over dl, reshuffling its DistributedSampler every epoch."""
    epoch = 0
    while True:
        if hasattr(dl.sampler, "set_epoch"):
            dl.sampler.set_epoch(epoch)
        for data in dl:
            yield data
        epoch += 1


# ---------------------------------------------------------------------------
# collectives with autograd (eager: they are only called from compiler-disabled code)
# ---------------------------------------------------------------------------
class _AllToAll(torch.autograd.Function):
    """x (size, ...): chunk j goes to rank j; chunk i of the output comes from rank i.
    The transpose of this exchange is the same exchange."""
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        x = x.contiguous()
        out = torch.empty_like(x)
        dist.all_to_all_single(out, x, group=group)
        return out

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        out = torch.empty_like(grad)
        dist.all_to_all_single(out, grad, group=ctx.group)
        return out, None


class _AllGather(torch.autograd.Function):
    """x (n, ...) -> (size * n, ...) in rank order. The backward reduce-scatters
    (sums) the gradient, in float32 so that the cp partial sums keep their precision."""
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        size = dist.get_world_size(group)
        x = x.contiguous()
        out = x.new_empty((size * x.shape[0], *x.shape[1:]))
        _all_gather_single(out, x, group=group)
        return out

    @staticmethod
    def backward(ctx, grad):
        size = dist.get_world_size(ctx.group)
        grad32 = grad.float().contiguous()
        out = grad32.new_empty((grad32.shape[0] // size, *grad32.shape[1:]))
        _reduce_scatter_single(out, grad32, op=dist.ReduceOp.SUM, group=ctx.group)
        return out.to(grad.dtype), None


# ---------------------------------------------------------------------------
# process groups and attention
# ---------------------------------------------------------------------------
class ContextParallel:
    """The (dp, cp) grid of the GPUs and this GPU's place in it.

    Needs the default process group (created by accelerate / torchrun). Ranks
    ``[k * size, (k + 1) * size)`` form CP group k, so with ``size`` dividing
    the GPUs per node the CP traffic stays inside a node (NVLink).

    Args:
        size: GPUs per CP group (1 = plain data parallel).
        backend: ``"ulysses"`` or ``"allgather"`` (see the module docstring).
    """
    def __init__(self, size, backend="ulysses"):
        if backend not in BACKENDS:
            raise ValueError(f"unknown context parallel backend {backend!r}, expected one of {BACKENDS}")
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        if size < 1 or world % size:
            raise ValueError(f"context parallel size {size} must divide the number of processes {world}")
        self.size = size
        self.backend = backend
        self.rank = rank % size        # rank inside the CP group
        self.dp_size = world // size   # number of CP groups
        self.dp_rank = rank // size    # index of this GPU's CP group
        self.group = None
        if size > 1:
            # DeviceMesh row-major layout: consecutive ranks share a CP group
            from torch.distributed.device_mesh import init_device_mesh
            self.mesh = init_device_mesh("cuda", (self.dp_size, size), mesh_dim_names=("dp", "cp"))
            self.group = self.mesh.get_group("cp")

    def __repr__(self):
        return f"ContextParallel(dp={self.dp_size}, cp={self.size}, backend={self.backend!r})"

    def train_dataloader(self, dataset, batch_size, collate_fn, patch_size, seed=0, **loader_kwargs):
        """Loader of this GPU: the samples of its DP group, sliced for its CP rank.

        batch_size is the global batch, split over the DP groups. All the GPUs of
        a group draw the same samples, and the global batch of step k is the same
        set of samples for every dp (and cp) size.
        """
        if batch_size % self.dp_size:
            raise ValueError(f"global batch {batch_size} must be divisible by the {self.dp_size} data parallel groups")
        sampler = DistributedSampler(dataset, num_replicas=self.dp_size, rank=self.dp_rank,
                                     shuffle=True, seed=seed, drop_last=True)
        if self.size > 1:
            collate_fn = ShardCollate(collate_fn, self.rank, self.size, patch_size)
        return DataLoader(dataset, batch_size=batch_size // self.dp_size, sampler=sampler, drop_last=True,
                          collate_fn=collate_fn, **loader_kwargs)

    @torch.compiler.disable
    def all_reduce(self, t):
        """Sum of t over the CP group (no gradient)."""
        t = t.detach().clone()
        if self.group is not None:
            dist.all_reduce(t, group=self.group)
        return t

    @torch.compiler.disable
    def global_sum(self, t):
        """Sum of t over all the GPUs (no gradient), e.g. the real points of the global batch."""
        t = t.detach().clone()
        if dist.is_initialized():
            dist.all_reduce(t)
        return t

    @torch.compiler.disable
    def prepare(self, token_mask):
        """Gather the token masks of the group once per forward: (B, S) local
        token mask -> :class:`CPAttention` shared by all the attention layers
        (None without CP: plain attention over the local tokens)."""
        if self.group is None:
            return None
        return CPAttention(self, token_mask)


class CPAttention:
    """Context parallel attention of one forward pass: the token masks of the
    group and the varlen metadata of FA4, computed once for all the layers."""
    def __init__(self, cp, token_mask):
        self.cp = cp
        P = cp.size
        B, S = token_mask.shape
        local = token_mask.contiguous()
        full = local.new_empty((P * B, S), dtype=torch.uint8)
        _all_gather_single(full, local.to(torch.uint8), group=cp.group)
        full = full.view(P, B, S).bool()  # rank-major: block s holds the tokens of rank s
        self.mask = local
        self.full_mask = full
        # real tokens of the gathered sequence, grouped per sample (as varlen expects),
        # as flat indices into the rank-major (P, B, S) layout
        b, s, t = full.permute(1, 0, 2).nonzero(as_tuple=True)
        self.full_idx = (s * B + b) * S + t
        lengths = full.sum(dim=(0, 2), dtype=torch.int32)
        self.cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
        self.max_seqlen = int(lengths.max())
        if cp.backend == "allgather":
            # the local queries
            self.local_idx = local.flatten().nonzero(as_tuple=True)[0]
            local_lengths = local.sum(dim=1, dtype=torch.int32)
            self.local_cu_seqlens = F.pad(local_lengths.cumsum(0, dtype=torch.int32), (1, 0))
            self.local_max_seqlen = int(local_lengths.max())

    def key_padding_mask(self):
        """(B, 1, 1, P * S) mask of the gathered keys, for SDPA."""
        P, B, S = self.full_mask.shape
        return self.full_mask.permute(1, 0, 2).reshape(B, 1, 1, P * S)

    @torch.compiler.disable
    def attention(self, q, k, v, use_flash=True):
        """q, k, v: (B, S, heads, head_dim) of the local tokens -> (B, S, heads, head_dim).

        Padded tokens get a zero output with FA4 (they are never used downstream)."""
        if self.cp.backend == "ulysses":
            return self._ulysses(q, k, v, use_flash)
        return self._allgather(q, k, v, use_flash)

    def _ulysses(self, q, k, v, use_flash):
        P, group = self.cp.size, self.cp.group
        B, S, H, d = q.shape
        if H % P:
            raise ValueError(f"ulysses context parallelism needs num_heads ({H}) divisible by cp ({P}); "
                             "use backend='allgather'")
        h = H // P
        # (B, S, 3, H, d) -> (P, B, S, 3, h, d): head block j goes to rank j
        qkv = torch.stack((q, k, v), dim=2).view(B, S, 3, P, h, d).permute(3, 0, 1, 2, 4, 5)
        qkv = _AllToAll.apply(qkv, group)  # (P, B, S, 3, h, d): block s holds the tokens of rank s
        if use_flash:
            from .attention import flash_attn_varlen_func
            packed = qkv.reshape(P * B * S, 3, h, d).index_select(0, self.full_idx)
            qp, kp, vp = (t.contiguous() for t in packed.unbind(1))
            out, *_ = flash_attn_varlen_func(
                qp, kp, vp,
                cu_seqlens_q=self.cu_seqlens, cu_seqlens_k=self.cu_seqlens,
                max_seqlen_q=self.max_seqlen, max_seqlen_k=self.max_seqlen,
            )
            out = out.new_zeros(P * B * S, h, d).index_copy(0, self.full_idx, out).view(P, B, S, h, d)
        else:
            qs, ks, vs = qkv.permute(3, 1, 4, 0, 2, 5).reshape(3, B, h, P * S, d).unbind(0)
            out = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=self.key_padding_mask())  # (B, h, P*S, d)
            out = out.view(B, h, P, S, d).permute(2, 0, 3, 1, 4)  # (P, B, S, h, d)
        out = _AllToAll.apply(out, group)  # (P, B, S, h, d): block s holds the heads of rank s
        return out.permute(1, 2, 0, 3, 4).reshape(B, S, H, d)

    def _allgather(self, q, k, v, use_flash):
        P, group = self.cp.size, self.cp.group
        B, S, H, d = q.shape
        kv = _AllGather.apply(torch.stack((k, v), dim=2), group)  # (P * B, S, 2, H, d), rank-major
        if use_flash:
            from .attention import flash_attn_varlen_func
            kp, vp = (t.contiguous() for t in kv.view(P * B * S, 2, H, d).index_select(0, self.full_idx).unbind(1))
            qp = q.reshape(B * S, H, d).index_select(0, self.local_idx)
            out, *_ = flash_attn_varlen_func(
                qp, kp, vp,
                cu_seqlens_q=self.local_cu_seqlens, cu_seqlens_k=self.cu_seqlens,
                max_seqlen_q=self.local_max_seqlen, max_seqlen_k=self.max_seqlen,
            )
            return out.new_zeros(B * S, H, d).index_copy(0, self.local_idx, out).view(B, S, H, d)
        ks, vs = kv.view(P, B, S, 2, H, d).permute(3, 1, 4, 0, 2, 5).reshape(2, B, H, P * S, d).unbind(0)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), ks, vs, attn_mask=self.key_padding_mask())
        return out.transpose(1, 2)
