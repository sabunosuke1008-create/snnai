from snnai.modules.language.long_context import LongContextLM


def test_long_context_lm_generates():
    vocab = "abcdefghijklmnopqrstuvwxyz "
    lm = LongContextLM(vocab, hidden_size=16, time_steps=5, capacity=8)
    text = lm.generate("hello", max_chars=5, store_every=2)
    assert isinstance(text, str)
    assert len(text) == 10
