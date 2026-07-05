"""Long-context language model using hippocampal memory."""
import torch

from snnai.modules.hippocampus.associative_memory import HippocampalMemory
from snnai.modules.language import CharTokenizer, SpikeEncoder
from snnai.modules.language.next_token import NextTokenSNN


class LongContextLM:
    """Language model that stores distant context in hippocampal memory.

    The model stores compressed context vectors at regular intervals and
    retrieves the most similar past context to bias next-token predictions.
    """

    def __init__(self, vocab, hidden_size=64, time_steps=10, capacity=32):
        self.tokenizer = CharTokenizer(vocab)
        self.encoder = SpikeEncoder(vocab_size=self.tokenizer.vocab_size, time_steps=time_steps)
        self.model = NextTokenSNN(vocab_size=self.tokenizer.vocab_size, hidden_size=hidden_size)
        self.memory = HippocampalMemory(dim=hidden_size, capacity=capacity)
        self.context_dim = hidden_size

    def _compress(self, spikes):
        """Compress spikes to a context vector using the model."""
        # Use hidden activations via a dummy forward to get spike output
        out = self.model(spikes)
        return out.mean(dim=(0, 1))  # (vocab_size,)

    def generate(self, prompt, max_chars=20, store_every=5):
        """Generate with episodic memory."""
        self.model.eval()
        text = prompt
        for i in range(max_chars):
            indices = torch.tensor([self.tokenizer.encode(text[-50:])])
            spikes = self.encoder(indices)
            context = self._compress(spikes)[:self.context_dim]
            if i % store_every == 0 and i > 0:
                self.memory.store(context, context)
            # Retrieve similar past context and add bias if memory is non-empty
            retrieved, _ = self.memory.retrieve(context, top_k=1)
            out = self.model(spikes)
            if retrieved.size(-2) > 0:
                bias = retrieved.squeeze(0).squeeze(0)[:self.tokenizer.vocab_size]
                out[0, -1, :bias.size(0)] += bias
            next_idx = int(torch.argmax(out[0, -1, :]).item())
            text += self.tokenizer.idx_to_char[next_idx]
        return text
