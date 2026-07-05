"""Minimal multimodal encoder fusing visual and audio-like features."""
import torch
import torch.nn as nn


class MultimodalEncoder(nn.Module):
    """Encode image-like and audio-like streams into a shared SNN feature space."""

    def __init__(self, image_size=16, audio_size=8, feature_size=16, hidden_size=32):
        super().__init__()
        self.image_encoder = nn.Linear(image_size, hidden_size)
        self.audio_encoder = nn.Linear(audio_size, hidden_size)
        self.fusion = nn.Linear(hidden_size * 2, feature_size)

    def forward(self, image, audio, time_steps=10):
        """Encode multimodal inputs.

        Parameters
        ----------
        image : torch.Tensor
            (batch, image_size).
        audio : torch.Tensor
            (batch, audio_size).
        time_steps : int
            Number of time steps to repeat.

        Returns
        -------
        torch.Tensor
            (time_steps, batch, feature_size).
        """
        vi = torch.relu(self.image_encoder(image))
        va = torch.relu(self.audio_encoder(audio))
        fused = torch.relu(self.fusion(torch.cat([vi, va], dim=-1)))
        return fused.unsqueeze(0).repeat(time_steps, 1, 1)
