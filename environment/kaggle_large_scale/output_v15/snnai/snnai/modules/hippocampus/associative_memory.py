"""Simple associative episodic memory inspired by the hippocampus."""
import torch
import torch.nn as nn


class HippocampalMemory(nn.Module):
    """Store and retrieve key-value memory patterns.

    Uses an associative Hopfield-like energy for retrieval. A query vector
    retrieves the stored value whose key is most similar.

    Parameters
    ----------
    dim : int
        Dimension of keys and values.
    capacity : int, optional
        Maximum number of stored episodes (default 64).
    """

    def __init__(self, dim, capacity=64):
        super().__init__()
        self.dim = dim
        self.capacity = capacity
        self.register_buffer("keys", torch.zeros(capacity, dim))
        self.register_buffer("values", torch.zeros(capacity, dim))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def store(self, key, value):
        """Store a key-value episode.

        Parameters
        ----------
        key : torch.Tensor
            Key tensor of shape (..., dim) or (dim,).
        value : torch.Tensor
            Value tensor of shape (..., dim) or (dim,).
        """
        if key.dim() == 1:
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
        n = min(key.size(0), self.capacity - self.count.item())
        if n <= 0:
            return
        start = self.count.item()
        end = start + n
        self.keys[start:end] = key[:n]
        self.values[start:end] = value[:n]
        self.count += n

    def retrieve(self, query, top_k=1):
        """Retrieve values most similar to the query.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor of shape (..., dim) or (dim,).
        top_k : int, optional
            Number of nearest neighbors to return.

        Returns
        -------
        torch.Tensor
            Retrieved values of shape (..., top_k, dim).
        torch.Tensor
            Similarity scores of shape (..., top_k).
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        active = self.keys[: self.count.item()]
        sim = torch.matmul(query, active.t())  # (..., count)
        top_k = min(top_k, self.count.item())
        if top_k == 0:
            shape = list(query.shape[:-1]) + [top_k, self.dim]
            return torch.zeros(*shape), torch.zeros(*query.shape[:-1], top_k)
        scores, indices = torch.topk(sim, k=top_k, dim=-1)
        retrieved = self.values[indices]  # (..., top_k, dim)
        if squeeze:
            retrieved = retrieved.squeeze(0)
            scores = scores.squeeze(0)
        return retrieved, scores

    def forward(self, query):
        """Convenience: retrieve top-1 value."""
        value, _ = self.retrieve(query, top_k=1)
        return value.squeeze(-2)
