"""Long-term memory bridge connecting SNNAI hippocampus and LLM context."""
import torch

from snnai.modules.hippocampus.associative_memory import HippocampalMemory
from snnai.modules.language import CharTokenizer
from snnai.modules.language.preprocess_pipeline import TextPreprocessPipeline


class MemoryBridge:
    """Bridge that stores user interactions and retrieves relevant context.

    Stores compressed representations of prompts/responses in hippocampal
    memory and retrieves them for future LLM prompts.
    """

    def __init__(self, vocab, feature_size=16, time_steps=10, capacity=64):
        self.tokenizer = CharTokenizer(vocab)
        self.pipeline = TextPreprocessPipeline(
            vocab_size=self.tokenizer.vocab_size, time_steps=time_steps, feature_size=feature_size
        )
        self.memory = HippocampalMemory(dim=feature_size, capacity=capacity)

    def store(self, text):
        """Store text in long-term memory."""
        features, _ = self._encode(text)
        context = features.mean(dim=(0, 2))
        self.memory.store(context, context)

    def retrieve(self, query, top_k=1):
        """Retrieve memory relevant to query."""
        features, _ = self._encode(query)
        context = features.mean(dim=(0, 2))
        retrieved, scores = self.memory.retrieve(context, top_k=top_k)
        return retrieved, scores

    def _encode(self, text):
        indices = torch.tensor([self.tokenizer.encode(text)])
        return self.pipeline(indices)
