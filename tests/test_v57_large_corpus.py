import torch
from snnai.modules.language import CharTokenizer
from snnai.modules.language.large_scale_lm import LargeScaleSNNLM
from snnai.benchmarks.large_corpus_trainer import LargeCorpusTrainer


def test_large_corpus_trainer():
    vocab = "abc "
    tokenizer = CharTokenizer(vocab)
    model = LargeScaleSNNLM(vocab_size=tokenizer.vocab_size, embed_dim=8, hidden_dim=16, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = LargeCorpusTrainer(model, optimizer, tokenizer)
    loss = trainer.train_step(["abc", "cba"], time_steps=3, seq_len=4)
    assert isinstance(loss, float)
