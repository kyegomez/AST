from collections import namedtuple
from functools import wraps

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from packaging import version
from torch import einsum, nn

# constants

Config = namedtuple(
    "Config", ["enable_flash", "enable_math", "enable_mem_efficient"]
)

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)

# main class


class Attend(nn.Module):
    def __init__(self, dropout=0.0, causal=False, flash=False):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.register_buffer("mask", None, persistent=False)

        self.flash = flash
        assert not (
            flash
            and version.parse(torch.__version__)
            < version.parse("2.0.0")
        ), (
            "in order to use flash attention, you must be using"
            " pytorch 2.0 or above"
        )

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(
            torch.device("cuda")
        )

        if (
            device_properties.major == 8
            and device_properties.minor == 0
        ):
            print_once(
                "A100 GPU detected, using flash attention if input"
                " tensor is on cuda"
            )
            self.cuda_config = Config(True, False, False)
        else:
            print_once(
                "Non-A100 GPU detected, using math or mem efficient"
                " attention if input tensor is on cuda"
            )
            self.cuda_config = Config(False, True, True)

    def flash_attn(self, q, k, v, mask=None):
        _, heads, q_len, _, k_len, is_cuda = (
            *q.shape,
            k.shape[-2],
            q.is_cuda,
        )

        k = repeat(k, "b ... -> b h ...", h=heads)
        v = repeat(v, "b ... -> b h ...", h=heads)

        causal = self.causal

        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            mask = mask.expand(-1, heads, q_len, -1)

            if causal:
                causal_mask = torch.ones(
                    (q_len, k_len), device=q.device, dtype=torch.bool
                ).triu(k_len - q_len + 1)
                mask = mask & ~causal_mask
                causal = False

        config = self.cuda_config if is_cuda else self.cpu_config

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=causal,
            )

        return out

    def forward(self, q, k, v, mask=None, attn_bias=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device = q.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        if self.flash:
            assert not exists(
                attn_bias
            ), "attention bias not supported for flash attention"
            return self.flash_attn(q, k, v, mask=mask)

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k) * scale

        # attention bias

        if exists(attn_bias):
            sim = sim + attn_bias

        # key padding mask

        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # causal mask

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones(
                (i, j), device=sim.device, dtype=torch.bool
            ).triu(j - i + 1)
            sim = sim.masked_fill(
                causal_mask, -torch.finfo(sim.dtype).max
            )

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        return out


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal=False,
        dim_head=64,
        dim_context=None,
        heads=8,
        norm_context=False,
        num_null_kv=0,
        dropout=0.1,
        scale=8,
        flash=False,
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        self.norm = nn.LayerNorm(dim)
        self.context_norm = (
            nn.LayerNorm(dim_context)
            if norm_context
            else nn.Identity()
        )

        self.attn_dropout = nn.Dropout(dropout)

        self.num_null_kv = num_null_kv
        self.null_kv = (
            nn.Parameter(torch.randn(2, num_null_kv, dim_head))
            if num_null_kv > 0
            else None
        )

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias=False)

        self.attend = Attend(
            flash=flash, dropout=dropout, causal=causal
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False), nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context=None,
        mask=None,
        attn_bias=None,
        prefix_context=None,
        prefix_context_mask=None,
        return_kv_cache=False,
        kv_cache=None,
    ):
        b, n, _, device = *x.shape, x.device

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        # take care of prefix-based self attention conditioning
        # make sure to either concat the to the self attention mask or lengthen it accordingly

        if exists(prefix_context):
            kv_input = torch.cat((prefix_context, kv_input), dim=-2)
            prefix_seq_len = prefix_context.shape[-2]

            if not exists(mask):
                mask = torch.ones(
                    (b, n), device=device, dtype=torch.bool
                )

            if exists(prefix_context_mask):
                mask = torch.cat((prefix_context_mask, mask), dim=-1)
            else:
                mask = F.pad(mask, (prefix_seq_len, 0), value=True)

            if exists(attn_bias):
                attn_bias = F.pad(
                    attn_bias, (prefix_seq_len, 0), value=0.0
                )

        # prenorm

        x = self.norm(x)

        # project for queries, keys, values

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)

        # kv cache

        if exists(kv_cache):
            ck, cv = kv_cache
            k = torch.cat((ck, k), dim=-2)
            v = torch.cat((cv, v), dim=-2)

        # store kv cache

        if return_kv_cache:
            kv_cache = torch.stack((k, v))

        # null key / values

        if self.num_null_kv > 0:
            null_k, null_v = repeat(
                self.null_kv, "kv n d -> kv b n d", b=b
            ).unbind(dim=0)
            k = torch.cat((null_k, k), dim=-2)
            v = torch.cat((null_v, v), dim=-2)

        # split for multi-headed attention

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)

        # handle mask and null key / value

        if exists(mask):
            mask = F.pad(mask, (self.num_null_kv, 0), value=True)

        # attention

        out = self.attend(q, k, v, attn_bias=attn_bias, mask=mask)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        if not return_kv_cache:
            return out

        return out, kv_cache
