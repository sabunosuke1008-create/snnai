import torch
from snnai.modules.language import CharTokenizer
from snnai.modules.language.large_scale_lm import LargeScaleSNNLM
from snnai.benchmarks.large_corpus_trainer import LargeCorpusTrainer, WarmupCosineSchedule


def test_large_corpus_trainer():
    vocab = "abc "
    tokenizer = CharTokenizer(vocab)
    model = LargeScaleSNNLM(vocab_size=tokenizer.vocab_size, embed_dim=8, hidden_dim=16, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = LargeCorpusTrainer(model, optimizer, tokenizer, val_ratio=0.0)
    history = trainer.train("abc abc abc cba cba cba", epochs=2, seq_len=4, batch_size=2, time_steps=3, save_path=None)
    assert 'train_loss' in history
    assert 'val_loss' in history
    assert len(history['train_loss']) == 2


def test_warmup_cosine_schedule():
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1e-3)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=5, total_steps=20, base_lr=1e-3, min_lr=1e-6)
    lrs = [scheduler.step() for _ in range(20)]
    assert lrs[0] < lrs[4]
    assert lrs[-1] <= 1e-3
    assert lrs[-1] >= 1e-6
