from snnai.modules.language.small_lm import SmallLanguageModel


def test_small_lm_forward():
    vocab = "abcdefghijklmnopqrstuvwxyz "
    lm = SmallLanguageModel(vocab, feature_size=8, hidden_size=16, time_steps=5)
    out = lm.forward("hello")
    assert out.shape[0] == 1
    assert out.shape[2] == lm.vocab_size


def test_small_lm_generate():
    vocab = "abcdefghijklmnopqrstuvwxyz "
    lm = SmallLanguageModel(vocab, feature_size=8, hidden_size=16, time_steps=5)
    generated = lm.generate("hi", max_chars=3)
    assert isinstance(generated, str)
    assert len(generated) == 5
