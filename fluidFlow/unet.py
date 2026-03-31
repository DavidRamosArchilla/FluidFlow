import math
from functools import partial

import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F


from einops import rearrange, repeat, pack, unpack

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def pack_one_with_inverse(x, pattern):
    packed, packed_shape = pack([x], pattern)

    def inverse(x, inverse_pattern = None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]

    return packed, inverse

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def project(x, y):
    x, inverse = pack_one_with_inverse(x, 'b *')
    y, _ = pack_one_with_inverse(y, 'b *')

    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = F.normalize(y, dim = -1)

    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthogonal = x - parallel

    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)


class Residual(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift, gate = scale_shift
            x = x * (scale + 1) + shift

        x = self.proj(x)
        x = self.act(x)
        if exists(scale_shift):
            x = x * gate
        return self.dropout(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, classes_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            # (scale, shift, gate) for block1 (scale, shift, gate) for block2
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim * 2 + dim_out * 4)
            # nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None
        self.dim_in = dim
        self.dim_out = dim_out
        self.block1 = Block(dim, dim_out, dropout)
        self.block2 = Block(dim_out, dim_out, dropout)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None, class_emb = None):

        # scale_shift_gate_block = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim = -1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1')
            # scale_shift_gate = torch.split(cond_emb, [self.dim_out, self.dim_out], dim=1)
            # first_chunk, second_chunk = cond_emb.chunk(2, dim=1)
            scale_shift_gate_block = torch.split(cond_emb, [self.dim_in, self.dim_in, self.dim_out, self.dim_out, self.dim_out, self.dim_out], dim=1)

        h = self.block1(x, scale_shift=scale_shift_gate_block[:3] if scale_shift_gate_block is not None else None)

        h = self.block2(h, scale_shift=scale_shift_gate_block[3:] if scale_shift_gate_block is not None else None)
        return h + self.res_conv(x)

class LinearAttention(Module):
    def __init__(self, dim, heads = 4, dim_head = 32, qknorm=False):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.qk_norm = qknorm
        self.q_norm = nn.RMSNorm(hidden_dim) if qknorm else identity
        self.k_norm = nn.RMSNorm(hidden_dim) if qknorm else identity

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        # qkv = self.to_qkv(x).chunk(3, dim = 1)
        # q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)
        q, k, v = self.to_qkv(x).chunk(3, dim = 1)
        q = rearrange(q, 'b c n -> b n c') 
        k = rearrange(k, 'b c n -> b n c')
        v = rearrange(v, 'b c n -> b n c')
        # q, k, v = qkv.unbind(1)  # Each is (B, N, C)
        
        # Apply normalization BEFORE reshaping into heads
        dtype = q.dtype
        if self.qk_norm:
            q = self.q_norm(q.to(self.q_norm.weight.dtype)).to(dtype)
            k = self.k_norm(k.to(self.k_norm.weight.dtype)).to(dtype) # (B, N, C)
        
        # Now reshape into multi-head format
        q = rearrange(q, 'b n (h d) -> b h d n', h=self.heads)  # (B, h, d, N)
        k = rearrange(k, 'b n (h d) -> b h d n', h=self.heads)  # (B, h, d, N)
        v = rearrange(v, 'b n (h d) -> b h d n', h=self.heads)  # (B, h, d, N)


        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class Attention(Module):
    def __init__(self, dim, heads = 4, dim_head = 32, qknorm=False):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.qk_norm = qknorm
        self.q_norm = nn.RMSNorm(hidden_dim) if qknorm else identity
        self.k_norm = nn.RMSNorm(hidden_dim) if qknorm else identity

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        # qkv = self.to_qkv(x).chunk(3, dim = 1)
        # q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)
        q, k, v = self.to_qkv(x).chunk(3, dim = 1)
        q = rearrange(q, 'b c n -> b n c') 
        k = rearrange(k, 'b c n -> b n c')
        v = rearrange(v, 'b c n -> b n c')
        dtype = q.dtype
        if self.qk_norm:
            q = self.q_norm(q.to(self.q_norm.weight.dtype)).to(dtype)
            k = self.k_norm(k.to(self.k_norm.weight.dtype)).to(dtype)
        TODO: n y c estan al reves, ver que hacer con esto porque esta funcionando mejor asi
        q, k, v = map(lambda t: rearrange(t, 'b n (h c) -> b h c n', h = self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v) # this outputs (b, h, c, n)
        out = rearrange(out, 'b h c n -> b (h c) n')

        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_q = nn.Conv1d(dim, hidden_dim, 1, bias=False)
        self.to_k = nn.Linear(context_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(context_dim, hidden_dim, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x, context):
        b, c, n = x.shape
        # context shape: (B, seq_len, context_dim)
        
        q = self.to_q(x)  # (B, hidden_dim, n)
        k = self.to_k(context)  # (B, seq_len, hidden_dim)
        v = self.to_v(context)  # (B, seq_len, hidden_dim)
        
        # Rearrange for multi-head attention
        q = rearrange(q, 'b (h d) n -> b h n d', h=self.heads)  # (B, heads, n, dim_head)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.heads)  # (B, heads, seq_len, dim_head)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.heads)  # (B, heads, seq_len, dim_head)

        q = q * self.scale

        # Attention: query positions attend to context sequence
        sim = einsum('b h i d, b h j d -> b h i j', q, k)  # (B, heads, n, seq_len)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # (B, heads, n, dim_head)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

# model

class Unet1D(Module):
    def __init__(
        self,
        dim,
        cond_dim, # number of conditioning classes
        cond_drop_prob=0.5,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        dropout = 0.,
        learn_sigma = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn=False,
        qknorm=False,
        cross_attn=False,
        self_condition = False,
    ):
        super().__init__()

        self.cond_drop_prob = cond_drop_prob
        # determine dimensions
        self.channels = channels
        self.cond_dim = cond_dim
        self.self_condition = self_condition
        self.learn_sigma = learn_sigma
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.null_classes_emb = nn.Parameter(torch.randn(cond_dim))

        classes_dim = dim * 4

        self.classes_mlp = nn.Sequential(
            nn.Linear(cond_dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )
        self.cross_attn = cross_attn

        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, classes_emb_dim=classes_dim, dropout = dropout)
        if cross_attn:
            outter_attention = partial(CrossAttention, context_dim=classes_dim, dim_head=attn_dim_head, heads=attn_heads, qknorm=qknorm)
        elif full_attn:
            outter_attention = partial(Attention, dim_head=attn_dim_head, heads=attn_heads, qknorm=qknorm)
        else:
            outter_attention = partial(LinearAttention, dim_head=attn_dim_head, heads=attn_heads, qknorm=qknorm)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                Residual(PreNorm(dim_in, outter_attention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head = attn_dim_head, heads = attn_heads, qknorm=qknorm)))
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                Residual(PreNorm(dim_out, outter_attention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learn_sigma else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv1d(init_dim, self.out_dim, 1)

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        rescaled_phi = 0.,
        remove_parallel_component = True,
        keep_parallel_frac = 0.,
        cfg_interval_start=0.,
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        update = logits - null_logits

        if remove_parallel_component:
            parallel, orthog = project(update, logits)
            update = orthog + parallel * keep_parallel_frac

        scaled_logits = logits + update * (cond_scale - 1.)
        if cfg_interval_start > 0:
            timestep = args[1][0] # args[1] is t
            if timestep < cfg_interval_start:
                scaled_logits = logits
        if rescaled_phi == 0.:
            return scaled_logits

        std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))
        interpolated_rescaled_logits = rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)

        return interpolated_rescaled_logits
    
    def forward_with_dpmsolver(self, x, timestep, y, mask=None, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, y) # mask
        return model_out.chunk(2, dim=1)[0] if self.learn_sigma else model_out


    def forward(self, x, time, classes, x_self_cond = None, cond_drop_prob=None, **kwargs):

        batch, device = x.shape[0], x.device
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)

            classes = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes,
                null_classes_emb
            )
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)
        c = self.classes_mlp(classes)
        x = self.init_conv(x)
        r = x.clone()

        if time is not None:
            t = self.time_mlp(time)
        else:
            t = None

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            attention_args = {"x": x}
            if self.cross_attn:
                attention_args["context"] = c
            x = attn(**attention_args)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)
        # adds pad to x (the fisrt tensor) so that the second tensor has the same length on the last dimension as the first one 
        maybe_pad = lambda x, h: F.pad(x, (0, h.shape[-1] - x.shape[-1])) if h.shape[-1] != x.shape[-1] else x

        for block1, block2, attn, upsample in self.ups:
            res_connection = h.pop()
            x = maybe_pad(x, res_connection) # check if x needs to be padded
            x = torch.cat((x, res_connection), dim = 1)
            x = block1(x, t, c)

            res_connection = h.pop()
            x = maybe_pad(x, res_connection)
            x = torch.cat((x, res_connection), dim = 1)
            x = block2(x, t, c)
            attention_args = {"x": x}
            if self.cross_attn:
                attention_args["context"] = c
            x = attn(**attention_args)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t, c)
        return self.final_conv(x)
