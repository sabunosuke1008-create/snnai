import torch
from snnai.modules.language.context_retention import ContextRetention


def test_context_retention_shape():
    cr = ContextRetention(dim=8, memory_slots=4, capacity=16)
    features = torch.randn(10, 2, 5, 8)
    wm_out, hp_out = cr(features)
    assert wm_out.shape == (2, 8)
    assert hp_out.shape == (2, 8)


def test_hippocampus_retrieves_stored():
    from snnai.modules.hippocampus import HippocampalMemory
    mem = HippocampalMemory(dim=8, capacity=8)
    key = torch.randn(1, 8)
    value = torch.randn(1, 8)
    mem.store(key, value)
    retrieved, scores = mem.retrieve(key, top_k=1)
    assert retrieved.shape == (1, 1, 8)
    assert scores.shape == (1, 1)
