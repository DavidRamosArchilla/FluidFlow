import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from timm.layers import trunc_normal_
try: 
    from flash_attn.cute import flash_attn_func
    is_flash_attn_available = True
    print("Flash attention 4 enabled ⚡")
except:
    is_flash_attn_available = False
    print("Flash attention 4 not found, using torch scaled_dot_product")

# Attention with rope and rmsnorm. Borrowed from https://github.dev/hustvl/LightningDiT/blob/main/models/lightningdit.py
# F4A was added too
class Attention(nn.Module):
    """
    Attention module of LightningDiT.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        proj_bias: bool = True,
        fused_attn: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
            
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qk_norm = qk_norm
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # 3, B, heads, N, head_dim
        q, k, v = qkv.unbind(0)
        dtype = q.dtype
        # q, k = self.q_norm(q), self.k_norm(k)
        # this is done this way to avoid dtype mismatch when using fp16/bf16
        if self.qk_norm:
            q = self.q_norm(q.to(self.q_norm.weight.dtype)).to(dtype)
            k = self.k_norm(k.to(self.k_norm.weight.dtype)).to(dtype)

        if rope is not None:
            q = rope(q)
            k = rope(k)
        # if i don't do this, it explodes when i compile the model, but only for some configurations 
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        if is_flash_attn_available:
            # FA4 expects (B, N, heads, head_dim) → permute from (B, heads, N, head_dim)
            q_fa = q.permute(0, 2, 1, 3)  # (B, N, heads, head_dim)
            k_fa = k.permute(0, 2, 1, 3)
            v_fa = v.permute(0, 2, 1, 3)
            x, *_ = flash_attn_func(
                q_fa, k_fa, v_fa,
                # dropout_p=self.attn_drop.p if self.training else 0.,
                causal=False,
            )
            # FA4 returns (B, N, heads, head_dim) → back to (B, heads, N, head_dim)
            x = x.permute(0, 2, 1, 3)
        elif self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LinearAttention(nn.Module):
    """
    A possible formulation can be found on https://arxiv.org/pdf/2503.16726
    """
    def __init__(self, dim, num_heads=4, qkv_bias=False, proj_bias=True, qk_norm=False, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, 'dimension must be divisible by number of heads'
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** -0.5
        self.heads = num_heads
        # TODO: temporary left qkv_bias unused
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.qk_norm = qk_norm
        self.q_norm = nn.RMSNorm(dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(dim) if qk_norm else nn.Identity()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim, bias=proj_bias),
            # nn.RMSNorm(dim),
        )
    
    def forward(self, x, rope=None):
        B, N, C = x.shape  # batch, sequence, channels
        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)  # Each is (B, N, C)
        
        # Apply normalization BEFORE reshaping into heads
        if self.qk_norm:
            q = self.q_norm(q.to(self.q_norm.weight.dtype))
            k = self.k_norm(k.to(self.k_norm.weight.dtype)) # (B, N, C)
        
        # apply rope first, since it assumes that sequence dimenstion (N) is in the position -2
        if rope is not None:
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)  # (B, h, d, N_q)
            k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)  # (B, h, d, N_kv)
            q = rope(q)
            k = rope(k)
            q = rearrange(q, 'b h n d -> b n (h d)')  # (B, N_q, C)
            k = rearrange(k, 'b h n d -> b n (h d)', h=self.heads)  # (B, N_kv, C)
        
        # Now reshape into multi-head format
        q = rearrange(q, 'b n (h d) -> b h d n', h=self.heads)  # (B, h, d, N)
        k = rearrange(k, 'b n (h d) -> b h d n', h=self.heads)  # (B, h, d, N)
        v = rearrange(v, 'b n (h d) -> b h d n', h=self.heads)  # (B, h, d, N)

        # use the relu approach https://export.arxiv.org/pdf/2410.10629
        q = F.relu(q, inplace=False)
        k = F.relu(k, inplace=False)

        eps = torch.finfo(q.dtype).eps        
        # Use matrix multiplication for normalization
        z = 1 / (k.sum(dim=-1, keepdim=True).transpose(-2, -1) @ q + eps)
        # k.sum(dim=-1, keepdim=True): (B, h, d, 1)
        # .transpose(-2, -1): (B, h, 1, d)
        # @ q: (B, h, 1, d) @ (B, h, d, N) = (B, h, 1, N)
        
        context = v @ k.transpose(-2, -1)  # (B, h, d, N) @ (B, h, N, d) = (B, h, d, d)
        out = context @ q  # (B, h, d, d) @ (B, h, d, N) = (B, h, d, N)
        out = (out * z)  # (B, h, d, N) * (B, h, 1, N) = (B, h, d, N)
        # out /= self.scale
        out = rearrange(out, 'b h d n -> b n (h d)')  # (B, N, C)
        return self.proj(out)


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    Adapted for 1D sequences.

    Args:
        dim (int): Number of input channels.
        window_size (int): The length of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., qk_norm=False):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # W
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define a parameter table of relative position bias
        # For 1D, the range of relative positions is [-(W-1), W-1], so we need 2*W - 1 buckets
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * window_size - 1, num_heads))  # 2*W-1, nH
        self.pos_bias_scale = nn.Parameter(torch.zeros(1)) 

        # Get pair-wise relative position index for each token inside the window
        coords = torch.arange(self.window_size)  # W
        relative_coords = coords[:, None] - coords[None, :]  # W, W
        relative_coords += self.window_size - 1  # shift to start from 0
        self.register_buffer("relative_position_index", relative_coords)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qk_norm = qk_norm
        self.q_norm = nn.RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(head_dim) if qk_norm else nn.Identity()

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C) 
            mask: (0/-inf) mask with shape of (num_windows, W, W) or None
        """
        B_, N, C = x.shape
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # (B_, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B_, N, 3, C)
        q, k, v = qkv.unbind(2)  # Each is (B, N, C)
        
        # Now reshape into multi-head format
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)  # (B, h, N, d)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)  # (B, h, N, d)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)  # (B, h, N, d)

        if self.qk_norm:
            q = self.q_norm(q.to(self.q_norm.weight.dtype))
            k = self.k_norm(k.to(self.k_norm.weight.dtype)) # (B, N, C)
            
        # Prepare relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1)  # W, W, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, W, W
        
        # Combine with window mask if present
        # if mask is not None:
        #     nW = mask.shape[0]
        #     # Expand relative_position_bias for all windows
        #     attn_mask = relative_position_bias.unsqueeze(0).expand(B_ // nW, -1, -1, -1)  # (B_//nW, nH, W, W)
        #     # Add window mask (broadcast across heads)
        #     attn_mask = attn_mask + mask.unsqueeze(0).unsqueeze(1)  # (B_//nW, nW, nH, W, W) -> (B_//nW, 1, 1, W, W)
        #     attn_mask = attn_mask.view(B_, self.num_heads, N, N)
        # else:
        #     # Just use relative position bias
        #     attn_mask = relative_position_bias.unsqueeze(0)  # (1, nH, W, W)
        relative_position_bias = relative_position_bias * torch.sigmoid(self.pos_bias_scale)
        attn_bias = relative_position_bias.expand(B_, -1, -1, -1)
        # Use PyTorch's fused attention
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        
        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# -- RoPE -- 
def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        max_seq_len=1024, # Set this large enough for your data (e.g. 1024 or 4096)
        theta = 10000,
    ):
        super().__init__()
        
        # 1. Generate the frequencies (1D only)
        # inv_freq shape: (dim // 2)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        
        # 2. Generate position indices: [0, 1, ..., max_seq_len-1]
        t = torch.arange(max_seq_len).float()
        
        # 3. Compute outer product: (max_seq_len, dim // 2)
        freqs = torch.outer(t, inv_freq)
        
        # 4. Repeat frequencies to match the specific "rotate_half" format
        # Your previous code used repeat(..., '... n -> ... (n r)', r=2)
        # This doubles the last dim so it matches the input shape
        freqs = repeat(freqs, 'n d -> n (d r)', r=2)
        
        # 5. Compute Sin and Cos
        freqs_cos = freqs.cos() # Shape (max_seq_len, dim)
        freqs_sin = freqs.sin() # Shape (max_seq_len, dim)

        # Register as buffers (so they are saved with state_dict but not trained)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        print(f'======== RoPE 1D initialized with shape {self.freqs_cos.shape} ========')

    def forward(self, t):
        # t shape: (Batch, Heads, Seq_Len, Dim)
        seq_len = t.shape[-2]
        
        # Slice the cached frequencies to the current sequence length
        # Reshape to (1, 1, Seq_Len, Dim) for broadcasting
        cos = self.freqs_cos[:seq_len].view(1, 1, seq_len, -1)
        sin = self.freqs_sin[:seq_len].view(1, 1, seq_len, -1)
        
        return t * cos + rotate_half(t) * sin