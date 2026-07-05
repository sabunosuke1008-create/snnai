import torch
from snnai.modules.language import CharTokenizer
from snnai.modules.language.preprocess_pipeline import TextPreprocessPipeline


def test_preprocess_pipeline_shape():
    tokenizer = CharTokenizer("abcde")
    pipeline = TextPreprocessPipeline(vocab_size=tokenizer.vocab_size, time_steps=10, feature_size=8)
    indices = torch.tensor([[0, 1, 2, 3, 4]])
    features, mask = pipeline(indices)
    assert features.shape == (10, 1, 5, 8)
    assert mask.shape == (1, 5)


def test_preprocess_pipeline_filters():
    tokenizer = CharTokenizer("abcde")
    pipeline = TextPreprocessPipeline(vocab_size=tokenizer.vocab_size, time_steps=10, feature_size=8)
    # Repeating the same token creates strong spikes
    indices = torch.tensor([[0, 0, 0, 3, 4]])
    _, mask = pipeline(indices)
    assert mask.shape == (1, 5)
