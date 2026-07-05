import torch
from snnai.modules.language import CharTokenizer
from snnai.modules.language.large_scale_lm import LargeScaleSNNLM
from snnai.benchmarks.transformer_challenge import run_challenge, challenge_report


def test_transformer_challenge():
    vocab = "abc "
    tokenizer = CharTokenizer(vocab)
    model = LargeScaleSNNLM(vocab_size=tokenizer.vocab_size, embed_dim=4, hidden_dim=8, num_layers=1)
    results = run_challenge(model, tokenizer, ["ab", "c"], max_chars=5)
    report = challenge_report(results)
    assert isinstance(report, str)
    assert "Challenge Report" in report
