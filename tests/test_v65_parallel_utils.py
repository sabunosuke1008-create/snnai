"""Tests for multi-GPU parallel utilities."""
import pytest
import torch
import torch.nn as nn

from snnai.benchmarks.parallel_utils import (
    PARALLEL_STRATEGIES,
    is_parallel,
    parallelize_model,
    unwrap_model,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def test_parallel_strategies_list():
    assert set(PARALLEL_STRATEGIES) == {'none', 'dp', 'ddp'}


def test_unwrap_non_wrapped_model():
    model = TinyModel()
    assert unwrap_model(model) is model
    assert not is_parallel(model)


def test_parallelize_none_returns_same_model():
    model = TinyModel()
    wrapped = parallelize_model(model, strategy='none')
    assert wrapped is model


def test_parallelize_invalid_strategy_raises():
    model = TinyModel()
    with pytest.raises(ValueError):
        parallelize_model(model, strategy='invalid')


def test_dp_wraps_model_when_cuda_available():
    model = TinyModel()
    # Even with no CUDA, DP construction should succeed (it checks device count).
    # If CUDA is available, it should wrap. If not, it should return unchanged
    # because the helper guards on torch.cuda.is_available().
    wrapped = parallelize_model(model, strategy='dp', dim=0)
    if torch.cuda.is_available():
        assert isinstance(wrapped, nn.DataParallel)
        assert is_parallel(wrapped)
        assert unwrap_model(wrapped) is model
    else:
        assert wrapped is model
        assert not is_parallel(wrapped)


def test_ddp_requires_process_group():
    model = TinyModel()
    with pytest.raises(RuntimeError):
        parallelize_model(model, strategy='ddp', dim=0)
