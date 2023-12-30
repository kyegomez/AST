from torch import Tensor, nn
from zeta.nn import SimpleFeedForward

from ast_torch.attention import Attention
from ast_torch.blocks import patch_split_overlap


class ASTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int,
        heads: int,
        depth: int,
        dropout: float = 0.1,
        ff_mult: int = 4,
        causal: bool = False,
        num_null_kv: int = 0,
        patch_size: int = 1,
        norm_context: bool = False,
        flash: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.depth = depth
        self.dropout = dropout
        self.ff_mult = ff_mult
        self.causal = causal
        self.num_null_kv = num_null_kv
        self.patch_size = patch_size
        self.norm_context = norm_context
        self.flash = flash

        self.to_patch = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_out = nn.Linear(dim_head * heads, dim, bias=False)
        self.ff_expansion_ration = dim * ff_mult
        self.attn = Attention(
            dim=dim,
            causal=causal,
            dim_head=dim_head,
            heads=heads,
            num_null_kv=num_null_kv,
            dropout=dropout,
            flash=flash,
            *args,
        )
        self.ffn = SimpleFeedForward(
            dim=dim,
            hidden_dim=self.ff_expansion_ration,
            dropout=dropout,
        )

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        x = patch_split_overlap(x, self.patch_size)
        x = self.to_patch(x)
        x = self.attn(x, mask=mask)
        x = self.to_out(x)
        x = self.ffn(x)
        return x
