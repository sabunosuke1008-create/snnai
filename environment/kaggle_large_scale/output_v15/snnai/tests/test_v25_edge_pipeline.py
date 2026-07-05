import torch
from snnai.modules.edge import EdgePipeline


def test_edge_pipeline_runs_end_to_end():
    pipeline = EdgePipeline(input_size=4, filter_threshold=1.0, event_threshold=5)
    x = torch.zeros(20, 2, 4)
    x[5:15, :, 0] = 2.0
    filtered, events, action = pipeline(x)
    assert filtered.shape == (20, 2, 4)
    assert events.shape == (2,)
    assert 0 <= action < 4


def test_edge_pipeline_input_size_mismatch():
    pipeline = EdgePipeline(input_size=3)
    x = torch.zeros(10, 1, 4)
    try:
        pipeline(x)
        assert False, "Expected ValueError"
    except ValueError:
        pass
