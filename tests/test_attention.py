"""FA4 attention must match PyTorch's SDPA, in the forward and in the gradients.

FA4 is a moving target: with 4.0.0b33, varlen attention with ``seqused_k``
gave the right forward but wrong gradients (4e6x too large), which only
showed up as a NaN loss after some training steps. Run on a GPU node after
updating flash-attn-4 / nvidia-cutlass-dsl:

    pytest tests/
"""
import pytest
import torch
import torch.nn.functional as F

import fluidFlow.attention as attention
from fluidFlow.attention import Attention

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and attention.is_flash_attn_available),
    reason="needs a GPU with FA4",
)

B, N, DIM, HEADS = 3, 1024, 256, 4


def _forward_backward(module, x, mask, grad_out, use_fa4, monkeypatch):
    """Output and gradients (input + parameters, flattened) with FA4 or SDPA."""
    monkeypatch.setattr(attention, "is_flash_attn_available", use_fa4)
    module.zero_grad()
    x = x.detach().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = module(x, mask=mask)
    (out.float() * grad_out).sum().backward()
    grads = torch.cat([x.grad.flatten()] + [p.grad.flatten() for p in module.parameters() if p.grad is not None])
    return out.detach().float(), grads


def _masks():
    lengths = torch.tensor([N, 700, 333])
    prefix = torch.arange(N) < lengths[:, None]  # padding at the end, as drivaernet_collate_fn
    scattered = torch.rand(B, N, generator=torch.Generator().manual_seed(0)) < 0.6
    return {"none": None, "prefix": prefix.cuda(), "scattered": scattered.cuda()}


@pytest.mark.parametrize("mask_kind", ["none", "prefix", "scattered"])
def test_fa4_matches_sdpa(mask_kind, monkeypatch):
    torch.manual_seed(0)
    module = Attention(DIM, num_heads=HEADS, qkv_bias=True, qk_norm=True).cuda()
    x = torch.randn(B, N, DIM, device="cuda")
    mask = _masks()[mask_kind]
    valid = torch.ones(B, N, dtype=torch.bool, device="cuda") if mask is None else mask
    # padded tokens are never used downstream, so they get no upstream gradient
    grad_out = torch.randn(B, N, DIM, device="cuda") * valid.unsqueeze(-1)

    out_fa4, grads_fa4 = _forward_backward(module, x, mask, grad_out, True, monkeypatch)
    out_sdpa, grads_sdpa = _forward_backward(module, x, mask, grad_out, False, monkeypatch)

    assert torch.isfinite(out_fa4).all() and torch.isfinite(grads_fa4).all()
    # forward: same output on the real tokens (bf16 tolerance)
    torch.testing.assert_close(out_fa4[valid], out_sdpa[valid], atol=2e-2, rtol=2e-2)
    # backward: same gradient direction and size
    cosine = F.cosine_similarity(grads_fa4, grads_sdpa, dim=0).item()
    norm_ratio = (grads_fa4.norm() / grads_sdpa.norm()).item()
    assert cosine > 0.999, f"gradient cosine FA4 vs SDPA {cosine:.4f}"
    assert abs(norm_ratio - 1) < 1e-2, f"gradient norm ratio FA4 / SDPA {norm_ratio:.4e}"
