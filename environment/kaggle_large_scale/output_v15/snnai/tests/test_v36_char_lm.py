from snnai.benchmarks.char_lm_trainer import CharLMTrainer


def test_char_lm_trains():
    corpus = "hello world hello snnai " * 20
    trainer = CharLMTrainer(corpus, hidden_size=16, time_steps=5, seq_len=10)
    history = trainer.train(epochs=5, batch_size=4, lr=1e-3)
    assert len(history) == 5
    assert history[-1] >= 0


def test_char_lm_generates():
    corpus = "hello world hello snnai " * 20
    trainer = CharLMTrainer(corpus, hidden_size=16, time_steps=5, seq_len=10)
    trainer.train(epochs=3, batch_size=4)
    text = trainer.generate("hello", max_chars=5)
    assert isinstance(text, str)
    assert len(text) == 10
