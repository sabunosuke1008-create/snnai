from snnai.benchmarks.text_generation_release import TextGenerator


def test_text_generator_runs():
    vocab = "abcdefghijklmnopqrstuvwxyz "
    gen = TextGenerator(vocab, embed_dim=8, hidden_dim=16, num_layers=1, time_steps=5)
    history = gen.train("hello world " * 30, epochs=3, batch_size=2, seq_len=10)
    assert len(history) == 3
    output = gen.generate("hello", max_chars=5)
    assert isinstance(output, str)
    assert len(output) == 10
