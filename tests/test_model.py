import torch
import pytest
from ast_torch.model import ASTransformer, SimpleFeedForward
from ast_torch.attention import Attention


@pytest.fixture
def model():
    return ASTransformer(
        dim=4, seqlen=16, dim_head=4, heads=4, depth=2, patch_size=4
    )


def test_to_patch_layer(model):
    assert isinstance(model.to_patch, torch.nn.Linear)
    assert model.to_patch.in_features == model.dim
    assert model.to_patch.out_features == model.dim_head * model.heads


def test_to_out_layer(model):
    assert isinstance(model.to_out, torch.nn.Linear)
    assert model.to_out.in_features == model.dim
    assert model.to_out.out_features == model.dim


def test_attention_layers(model):
    assert isinstance(model.attn_layers, torch.nn.ModuleList)
    assert len(model.attn_layers) == model.depth
    for layer in model.attn_layers:
        assert isinstance(layer, Attention)


def test_feed_forward_layers(model):
    assert isinstance(model.ffn_layers, torch.nn.ModuleList)
    assert len(model.ffn_layers) == model.depth
    for layer in model.ffn_layers:
        assert isinstance(layer, SimpleFeedForward)


def test_forward_output(model):
    x = torch.randn(1, 16)
    output = model(x)
    assert output.shape == (1, model.dim)
