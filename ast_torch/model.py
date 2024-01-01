from torch import Tensor, nn
from zeta.nn import PositionalEmbedding, SimpleFeedForward

from ast_torch.attention import Attention
from ast_torch.blocks import patch_split_overlap


class ASTransformer(nn.Module):
    """
    ASTransformer is a transformer-based model for AST (Abstract Syntax Tree) processing.

    Args:
        dim (int): The dimension of the input and output tensors.
        dim_head (int): The dimension of each attention head.
        heads (int): The number of attention heads.
        depth (int): The number of transformer layers.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        ff_mult (int, optional): The expansion ratio for the feed-forward network. Defaults to 4.
        causal (bool, optional): Whether to use causal attention. Defaults to False.
        num_null_kv (int, optional): The number of null key-value pairs. Defaults to 0.
        patch_size (int, optional): The patch size for patching the input. Defaults to 1.
        norm_context (bool, optional): Whether to normalize the context. Defaults to False.
        flash (bool, optional): Whether to use FLASH attention. Defaults to False.

    Attributes:
        dim (int): The dimension of the input and output tensors.
        dim_head (int): The dimension of each attention head.
        heads (int): The number of attention heads.
        depth (int): The number of transformer layers.
        dropout (float): The dropout rate.
        ff_mult (int): The expansion ratio for the feed-forward network.
        causal (bool): Whether to use causal attention.
        num_null_kv (int): The number of null key-value pairs.
        patch_size (int): The patch size for patching the input.
        norm_context (bool): Whether to normalize the context.
        flash (bool): Whether to use FLASH attention.
        to_patch (nn.Linear): Linear layer for patching the input.
        to_out (nn.Linear): Linear layer for projecting the output.
        to_1d_embeddings (nn.Linear): Linear layer for 1D embeddings.
        ff_expansion_ration (int): Expansion ratio for the feed-forward network.
        attn (Attention): Attention module.
        ffn (SimpleFeedForward): Feed-forward network module.
        attn_layers (nn.ModuleList): List of attention layers.
        ffn_layers (nn.ModuleList): List of feed-forward network layers.

    """

    def __init__(
        self,
        dim: int,
        seqlen: int,
        dim_head: int,
        heads: int,
        depth: int,
        dropout: float = 0.1,
        ff_mult: int = 4,
        causal: bool = False,
        num_null_kv: int = 0,
        patch_size: int = 1,
        flash: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.seqlen = seqlen
        self.dim_head = dim_head
        self.heads = heads
        self.depth = depth
        self.dropout = dropout
        self.ff_mult = ff_mult
        self.causal = causal
        self.num_null_kv = num_null_kv
        self.patch_size = patch_size
        self.flash = flash

        self.to_out = nn.Linear(dim, dim, bias=False)
        self.to_1d_embeddings = nn.Linear(dim, dim)
        self.ff_expansion_ration = dim * ff_mult
        
        # Layers
        self.attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])

        for _ in range(self.depth):
            self.attn_layers.append(
                Attention(
                    dim=dim,
                    causal=causal,
                    dim_head=dim_head,
                    heads=heads,
                    num_null_kv=num_null_kv,
                    dropout=dropout,
                    flash=flash,
                    *args,
                )
            )
            self.ffn_layers.append(
                SimpleFeedForward(
                    dim=dim,
                    hidden_dim=self.ff_expansion_ration,
                    dropout=dropout,
                )
            )

        self.pos_emb = PositionalEmbedding(seqlen, dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the ASTransformer.

        Args:
            x (Tensor): The input tensor.
            mask (Tensor, optional): The mask tensor. Defaults to None.

        Returns:
            Tensor: The output tensor.

        """
        # Patching then embedding
        x = patch_split_overlap(x, self.patch_size)
        x = self.to_1d_embeddings(x)
        x = self.pos_emb(x)

        # For i in depth then do attn and ffn
        for attn_block, ffn_block in zip(
            self.attn_layers, self.ffn_layers
        ):
            x = attn_block(x) + x
            x = ffn_block(x) + x

        # Projecting to output
        x = self.to_out(x)

        return x
