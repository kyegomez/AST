import pytest
import torch
from ast_torch.model import ASTransformer

@pytest.fixture
def model():
    dim = 512
    seqlen = 100
    dim_head = 64
    heads = 8
    depth = 6
    dropout = 0.1
    ff_mult = 4
    causal = False
    num_null_kv = 0
    patch_size = 1
    norm_context = False
    flash = False
    return ASTransformer(
        dim=dim,
        seqlen=seqlen,
        dim_head=dim_head,
        heads=heads,
        depth=depth,
        dropout=dropout,
        ff_mult=ff_mult,
        causal=causal,
        num_null_kv=num_null_kv,
        patch_size=patch_size,
        norm_context=norm_context,
        flash=flash
    )

def test_initialization(model):
    assert model.dim == 512
    assert model.seqlen == 100
    assert model.dim_head == 64
    assert model.heads == 8
    assert model.depth == 6
    assert model.dropout == 0.1
    assert model.ff_mult == 4
    assert model.causal == False
    assert model.num_null_kv == 0
    assert model.patch_size == 1
    assert model.norm_context == False
    assert model.flash == False

def test_forward(model):
    x = torch.rand(1, model.seqlen, model.dim)
    output = model(x)
    assert isinstance(output, torch.Tensor)

def test_output_shape(model):
    x = torch.rand(1, model.seqlen, model.dim)
    output = model(x)
    assert output.shape == (1, model.seqlen, model.dim)