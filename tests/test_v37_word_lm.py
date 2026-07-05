from snnai.modules.language.word_lm import WordLMTrainer


def test_word_lm_trains():
    texts = ["hello world", "hello snnai", "event driven ai", "low power network"]
    trainer = WordLMTrainer(texts, hidden_size=16, time_steps=5, seq_len=4)
    history = trainer.train(epochs=5, batch_size=2, lr=1e-3)
    assert len(history) == 5


def test_word_lm_generates():
    texts = ["hello world", "hello snnai", "event driven ai", "low power network"]
    trainer = WordLMTrainer(texts, hidden_size=16, time_steps=5, seq_len=4)
    trainer.train(epochs=3, batch_size=2)
    output = trainer.generate("hello", max_words=2)
    assert isinstance(output, str)
