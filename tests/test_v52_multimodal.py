import torch
from snnai.modules.multimodal import MultimodalEncoder


def test_multimodal_encoder():
    encoder = MultimodalEncoder(image_size=8, audio_size=4, feature_size=4, hidden_size=8)
    image = torch.randn(2, 8)
    audio = torch.randn(2, 4)
    out = encoder(image, audio, time_steps=5)
    assert out.shape == (5, 2, 4)
