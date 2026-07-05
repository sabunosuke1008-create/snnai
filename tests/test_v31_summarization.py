from snnai.benchmarks.summarization import ExtractiveSummarizer


def test_summarizer_returns_subset():
    vocab = "abcdefghijklmnopqrstuvwxyz "
    summarizer = ExtractiveSummarizer(vocab, top_k=3, time_steps=5, feature_size=4)
    text = "hello world"
    summary = summarizer.summarize(text)
    assert isinstance(summary, str)
    assert len(summary) <= 3
