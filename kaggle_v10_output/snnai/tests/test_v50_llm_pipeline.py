from snnai.modules.llm_bridge.pipeline import LLMCollaborationPipeline


def test_llm_collaboration_pipeline():
    vocab = "abcdefghijklmnopqrstuvwxyz "
    pipeline = LLMCollaborationPipeline(vocab, feature_size=4, hidden_size=8)
    output, meta = pipeline.run("hello world", max_chars=15)
    assert isinstance(output, str)
    assert "preprocess" in meta
    assert "memory_used" in meta
