"""Minimal byte-pair encoding (BPE) tokenizer for SNNAI."""
import re


class BPETokenizer:
    """Minimal BPE tokenizer.

    Parameters
    ----------
    texts : list[str]
        Training texts.
    vocab_size : int, optional
        Target vocabulary size (default 50).
    """

    def __init__(self, texts, vocab_size=50):
        # Pre-tokenize into words with end-of-word symbol </w>
        word_freqs = {}
        for t in texts:
            for w in t.lower().split():
                w = " ".join(list(w)) + " </w>"
                word_freqs[w] = word_freqs.get(w, 0) + 1

        # Initial vocab: characters + </w>
        chars = set()
        for w in word_freqs:
            chars.update(w.split())
        self.vocab = sorted(chars)

        self.merges = []
        while len(self.vocab) < vocab_size:
            pairs = {}
            for word, freq in word_freqs.items():
                tokens = word.split()
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pairs[pair] = pairs.get(pair, 0) + freq
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            merged = best[0] + best[1]
            if merged not in self.vocab:
                self.vocab.append(merged)
            # Update word representations
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word_freqs[self._apply_merge(word, best)] = freq
            word_freqs = new_word_freqs

        self.token_to_idx = {t: i for i, t in enumerate(self.vocab)}
        self.idx_to_token = {i: t for i, t in enumerate(self.vocab)}

    def _apply_merge(self, word, pair):
        tokens = word.split()
        merged = pair[0] + pair[1]
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return " ".join(new_tokens)

    def encode(self, text):
        """Encode text to token indices."""
        out = []
        for w in text.lower().split():
            word = " ".join(list(w)) + " </w>"
            for pair in self.merges:
                word = self._apply_merge(word, pair)
            for t in word.split():
                out.append(self.token_to_idx.get(t, 0))
        return out

    def decode(self, indices):
        """Decode token indices to text."""
        tokens = [self.idx_to_token.get(i, "<unk>") for i in indices]
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        return text.strip()
