"""Extractive summarization benchmark using SNNAI text modules."""
import torch

from snnai.modules.language import CharTokenizer
from snnai.modules.language.preprocess_pipeline import TextPreprocessPipeline


class ExtractiveSummarizer:
    """Simple extractive summarizer based on token saliency.

    Tokens with higher spike activity after preprocessing are considered more
    important and selected for the summary.
    """

    def __init__(self, vocab, top_k=3, time_steps=10, feature_size=8):
        self.tokenizer = CharTokenizer(vocab)
        self.top_k = top_k
        self.pipeline = TextPreprocessPipeline(
            vocab_size=self.tokenizer.vocab_size,
            time_steps=time_steps,
            feature_size=feature_size,
        )

    def summarize(self, text):
        """Return the top-k most salient characters as a summary.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        str
            Extracted summary string.
        """
        indices = torch.tensor([self.tokenizer.encode(text)])
        features, mask = self.pipeline(indices)
        # Saliency: sum of spike counts over time and feature dimensions
        saliency = features.sum(dim=(0, 3)).squeeze(0)  # (seq_len,)
        seq_len = min(self.top_k, saliency.numel())
        top_indices = torch.topk(saliency, k=seq_len).indices.sort().values
        summary = "".join(self.tokenizer.idx_to_char[int(i)] for i in top_indices)
        return summary
