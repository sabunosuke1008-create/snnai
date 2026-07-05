import torch
from snnai.benchmarks.finetune import FineTuner
from snnai.modules.language import CharTokenizer, SpikeEncoder
from snnai.modules.language.next_token import NextTokenSNN


def test_finetuner():
    corpus = "hello world hello snnai " * 20
    tokenizer = CharTokenizer(corpus)
    encoder = SpikeEncoder(vocab_size=tokenizer.vocab_size, time_steps=5)
    model = NextTokenSNN(vocab_size=tokenizer.vocab_size, hidden_size=16)
    tuner = FineTuner(model, tokenizer, encoder)
    history = tuner.finetune(corpus, epochs=3, batch_size=2, seq_len=10)
    assert len(history) == 3
