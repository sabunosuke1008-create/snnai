from snnai.modules.language.bpe_tokenizer import BPETokenizer


def test_bpe_roundtrip():
    texts = ["hello world", "hello snnai", "event driven", "low power"]
    tokenizer = BPETokenizer(texts, vocab_size=30)
    encoded = tokenizer.encode("hello")
    decoded = tokenizer.decode(encoded)
    assert isinstance(encoded, list)
    assert isinstance(decoded, str)


def test_bpe_vocab_grows():
    texts = ["hello world", "hello snnai", "event driven", "low power"]
    tokenizer = BPETokenizer(texts, vocab_size=30)
    assert len(tokenizer.vocab) <= 30
    assert len(tokenizer.vocab) > 0
