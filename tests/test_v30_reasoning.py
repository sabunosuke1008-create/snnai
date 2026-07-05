import torch
from snnai.modules.language.reasoning import ReasoningModule


def test_reasoning_module_shape():
    module = ReasoningModule(input_dim=8, hidden_dim=16, output_dim=4)
    x = torch.randn(2, 8)
    out = module(x, time_steps=10)
    assert out.shape == (2, 4)
