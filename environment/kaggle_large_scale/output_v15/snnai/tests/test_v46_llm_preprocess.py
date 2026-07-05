from snnai.modules.llm_bridge.preprocess import LLMPreProcessor


def test_llm_preprocessor():
    vocab = "abcdefghijklmnopqrstuvwxyz "
    processor = LLMPreProcessor(vocab, feature_size=4, time_steps=5, top_k_ratio=0.6)
    compressed, meta = processor.process("hello world snnai")
    assert isinstance(compressed, str)
    assert "kept_ratio" in meta
    assert 0 <= meta["kept_ratio"] <= 1
