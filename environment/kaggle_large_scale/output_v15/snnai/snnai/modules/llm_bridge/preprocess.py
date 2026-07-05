"""SNNAI preprocessor for LLM input."""
import torch

from snnai.modules.language.preprocess_pipeline import TextPreprocessPipeline


class LLMPreProcessor:
    """Filter and summarize user input before sending to LLM.

    Uses the bio-inspired preprocessing pipeline (C. elegans + insect)
    to extract salient tokens and compress the prompt.
    """

    def __init__(self, vocab, feature_size=16, time_steps=10, top_k_ratio=0.5):
        self.pipeline = TextPreprocessPipeline(
            vocab_size=len(set(vocab)), time_steps=time_steps, feature_size=feature_size
        )
        self.vocab = vocab
        self.top_k_ratio = top_k_ratio

    def process(self, text):
        """Return a compressed prompt for the LLM.

        Parameters
        ----------
        text : str
            Raw user input.

        Returns
        -------
        str
            Compressed prompt.
        dict
            Preprocessing metadata.
        """
        from snnai.modules.language import CharTokenizer
        tokenizer = CharTokenizer(self.vocab)
        indices = torch.tensor([tokenizer.encode(text)])
        _, mask = self.pipeline(indices)
        seq_len = mask.size(1)
        top_k = max(1, int(seq_len * self.top_k_ratio))
        kept_positions = torch.topk(mask.float().sum(dim=0), k=min(top_k, seq_len)).indices.sort().values
        kept_chars = [text[i] for i in kept_positions.tolist() if i < len(text)]
        compressed = "".join(kept_chars)
        return compressed, {"kept_ratio": len(kept_chars) / max(len(text), 1)}
