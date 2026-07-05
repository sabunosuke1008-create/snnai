from snnai.benchmarks.translation import SimpleTranslator


def test_translator_outputs_string():
    vocab = "abcdefghijklmnopqrstuvwxyz "
    translator = SimpleTranslator(vocab, hidden_size=16, time_steps=5)
    output = translator.translate("hello")
    assert isinstance(output, str)
    assert len(output) == 6
