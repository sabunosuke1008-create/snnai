"""User adaptation / personalization module for SNNAI."""
import torch
import torch.nn as nn


class UserAdapter(nn.Module):
    """Maintain a low-rank user embedding and add it to SNN outputs."""

    def __init__(self, feature_size=16, num_users=4, rank=4):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, rank)
        self.adapter = nn.Linear(rank, feature_size, bias=False)

    def forward(self, features, user_id):
        """Add personalized bias to features.

        Parameters
        ----------
        features : torch.Tensor
            (time_steps, batch, feature_size).
        user_id : int or torch.Tensor
            User identifier.

        Returns
        -------
        torch.Tensor
            Personalized features of same shape.
        """
        if isinstance(user_id, int):
            user_id = torch.tensor([user_id], dtype=torch.long, device=features.device)
        emb = self.user_embeddings(user_id)
        bias = self.adapter(emb)
        return features + bias.unsqueeze(0)
