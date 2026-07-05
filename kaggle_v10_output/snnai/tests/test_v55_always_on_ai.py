import torch
from snnai.benchmarks.always_on_ai_v1 import AlwaysOnAIV1


def test_always_on_ai_v1():
    vocab = "abcdefghijklmnopqrstuvwxyz "
    ai = AlwaysOnAIV1(vocab, sensor_input_size=4)
    stream = torch.randn(40, 1, 4) * 0.2
    stream[20:30, :, 0] += 2.0
    result = ai.run_cycle(stream, user_query="status")
    assert isinstance(result["alert"], bool)

