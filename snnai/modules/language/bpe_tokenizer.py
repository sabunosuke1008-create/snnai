"""Minimal byte-pair encoding (BPE) tokenizer for SNNAI."""
from collections import Counter, defaultdict


class BPETokenizer:
    """Minimal BPE tokenizer.

    Parameters
    ----------
    texts : list[str]
        Training texts.
    vocab_size : int, optional
        Target vocabulary size (default 50).
    max_train_bytes : int, optional
        Maximum number of bytes to use for training the BPE merge table.
        Useful when the full corpus is very large and training would be too
        slow. If None, the full corpus is used.
    """

    def __init__(self, texts, vocab_size=50, max_train_bytes=None):
        train_text = ''.join(texts)
        if max_train_bytes is not None and len(train_text) > max_train_bytes:
            train_text = train_text[:max_train_bytes]

        # Build word frequencies as tuples of characters ending with </w>.
        raw_word_freqs = Counter()
        for w in train_text.lower().split():
            raw_word_freqs[tuple(list(w)) + ('</w>',)] += 1

        # Initial vocab: characters + </w>
        chars = set()
        for tokens in raw_word_freqs:
            chars.update(tokens)
        self.vocab = sorted(chars)

        # Indexed word storage: word_id -> (tokens, freq)
        word_freqs = {}
        for idx, (tokens, freq) in enumerate(raw_word_freqs.items()):
            word_freqs[idx] = [list(tokens), freq]

        # Pair frequencies and inverted index.
        pair_freqs = Counter()
        pair_to_words = defaultdict(set)
        for idx, (tokens, freq) in word_freqs.items():
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freqs[pair] += freq
                pair_to_words[pair].add(idx)

        self.merges = []
        while len(self.vocab) < vocab_size and pair_freqs:
            best = max(pair_freqs, key=pair_freqs.get)
            self.merges.append(best)
            merged = best[0] + best[1]
            if merged not in self.vocab:
                self.vocab.append(merged)

            # Update only words that contain the merged pair.
            affected = list(pair_to_words.get(best, set()))
            for idx in affected:
                tokens, freq = word_freqs[idx]
                new_tokens = self._apply_merge(tokens, best)
                if new_tokens == tokens:
                    continue

                # Decrement old pair frequencies.
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_freqs[pair] -= freq
                    if pair_freqs[pair] <= 0:
                        del pair_freqs[pair]
                    pair_to_words[pair].discard(idx)

                # Increment new pair frequencies.
                for i in range(len(new_tokens) - 1):
                    pair = (new_tokens[i], new_tokens[i + 1])
                    pair_freqs[pair] += freq
                    pair_to_words[pair].add(idx)

                word_freqs[idx][0] = new_tokens

        self.token_to_idx = {t: i for i, t in enumerate(self.vocab)}
        self.idx_to_token = {i: t for i, t in enumerate(self.vocab)}
        # Maintain a CharTokenizer-compatible interface for pipelines that use
        # vocab_size / char_to_idx / idx_to_char.
        self.vocab_size = len(self.vocab)
        self.char_to_idx = self.token_to_idx
        self.idx_to_char = self.idx_to_token

    @staticmethod
    def _apply_merge(tokens, pair):
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
        return new_tokens

    def encode(self, text):
        """Encode text to token indices."""
        out = []
        for w in text.lower().split():
            tokens = list(w) + ['</w>']
            for pair in self.merges:
                tokens = self._apply_merge(tokens, pair)
            for t in tokens:
                out.append(self.token_to_idx.get(t, 0))
        return out

    def decode(self, indices):
        """Decode token indices to text."""
        tokens = [self.idx_to_token.get(i, "<unk>") for i in indices]
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        return text.strip()
