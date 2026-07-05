"""SNNAI + LLM collaboration pipeline."""
import torch

from snnai.modules.language import CharTokenizer
from snnai.modules.llm_bridge.memory_bridge import MemoryBridge
from snnai.modules.llm_bridge.postprocess import LLMPostProcessor
from snnai.modules.llm_bridge.preprocess import LLMPreProcessor


class LLMCollaborationPipeline:
    """Pipeline: SNNAI preprocess -> LLM stub -> memory -> postprocess.

    The LLM is represented by a simple deterministic stub in this concept.
    In production, this calls an external LLM API.
    """

    def __init__(self, vocab, feature_size=8, hidden_size=16):
        self.vocab = vocab
        self.tokenizer = CharTokenizer(vocab)
        self.preprocessor = LLMPreProcessor(vocab, feature_size=feature_size, time_steps=5)
        self.memory = MemoryBridge(vocab, feature_size=feature_size, time_steps=5, capacity=16)
        self.postprocessor = LLMPostProcessor(vocab_size=self.tokenizer.vocab_size, feature_size=feature_size,
                                              hidden_size=hidden_size)

    def llm_stub(self, prompt, max_chars=20):
        """Simple deterministic LLM stub: echo prompt with minor transform."""
        return (prompt + " response").strip()[:max_chars]

    def run(self, user_input, max_chars=20):
        """Run the full collaboration pipeline."""
        compressed, meta = self.preprocessor.process(user_input)
        memory_context, _ = self.memory.retrieve(compressed, top_k=1)
        if memory_context.size(-2) > 0:
            # Add memory context string placeholder
            prompt = compressed + " [ctx]"
        else:
            prompt = compressed
        draft = self.llm_stub(prompt, max_chars=max_chars)
        self.memory.store(user_input + " " + draft)
        output = self.postprocessor.process(draft, self.tokenizer)
        return output, {"preprocess": meta, "memory_used": memory_context.size(-2) > 0}
