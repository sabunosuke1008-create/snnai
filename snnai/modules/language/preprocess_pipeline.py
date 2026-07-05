"""Text preprocessing pipeline using bio-inspired modules."""
import torch
import torch.nn as nn
import snntorch as snn

from snnai.modules.language.tokenizer import SpikeEncoder


class ReflexFilter(nn.Module):
    """C. elegans-like reflex filter that removes low-saliency tokens.

    A token is kept only if its spike activity exceeds a threshold.

    Parameters
    ----------
    threshold : float, optional
        Minimum total spike count to keep a token (default 2.0).
    """

    def __init__(self, threshold=2.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, spikes):
        """Filter tokens.

        Parameters
        ----------
        spikes : torch.Tensor
            Spike tensor of shape (time, batch, seq_len, vocab_size).

        Returns
        -------
        torch.Tensor
            Filtered spikes of the same shape. Low-saliency tokens are zeroed.
        mask : torch.Tensor
            Boolean mask of shape (batch, seq_len) indicating kept tokens.
        """
        counts = spikes.sum(dim=0).sum(dim=2)
        mask = counts > self.threshold
        mask_time = mask.unsqueeze(0).unsqueeze(3)
        return spikes * mask_time.float(), mask


class FeatureEncoder(nn.Module):
    """Insect-like feature encoder that transforms token spikes into features.

    Uses a small SNN layer to extract local patterns from token sequences.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    feature_size : int, optional
        Output feature size (default 32).
    """

    def __init__(self, vocab_size, feature_size=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.feature_size = feature_size
        self.fc = nn.Linear(vocab_size, feature_size, bias=False)
        self.lif = snn.Leaky(beta=0.9, threshold=1.0, learn_threshold=False)

    def forward(self, spikes):
        """Encode token spikes to features.

        Parameters
        ----------
        spikes : torch.Tensor
            Spike tensor of shape (time, batch, seq_len, vocab_size).

        Returns
        -------
        torch.Tensor
            Feature tensor of shape (time, batch, seq_len, feature_size).
        """
        time_steps, batch_size, seq_len, _ = spikes.shape
        mem = torch.zeros(batch_size, seq_len, self.feature_size)
        features = []
        for t in range(time_steps):
            cur = self.fc(spikes[t])
            spk, mem = self.lif(cur, mem)
            features.append(spk)
        return torch.stack(features)


class TextPreprocessPipeline(nn.Module):
    """Bio-inspired text preprocessing: filter + encode.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    time_steps : int, optional
        Spike time steps (default 20).
    feature_size : int, optional
        Feature size (default 32).
    """

    def __init__(self, vocab_size, time_steps=20, feature_size=32):
        super().__init__()
        self.encoder = SpikeEncoder(vocab_size=vocab_size, time_steps=time_steps)
        self.reflex_filter = ReflexFilter(threshold=2.0)
        self.feature_encoder = FeatureEncoder(vocab_size=vocab_size, feature_size=feature_size)

    def forward(self, indices):
        """Preprocess token indices.

        Parameters
        ----------
        indices : torch.Tensor
            Token indices of shape (batch, seq_len).

        Returns
        -------
        features : torch.Tensor
            Encoded features of shape (time, batch, seq_len, feature_size).
        mask : torch.Tensor
            Kept-token mask of shape (batch, seq_len).
        """
        spikes = self.encoder(indices)
        filtered, mask = self.reflex_filter(spikes)
        features = self.feature_encoder(filtered)
        return features, mask
