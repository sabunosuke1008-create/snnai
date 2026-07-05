from snnai.modules.llm_bridge.memory_bridge import MemoryBridge


def test_memory_bridge_store_retrieve():
    vocab = "abcdefghijklmnopqrstuvwxyz "
    bridge = MemoryBridge(vocab, feature_size=4, time_steps=5, capacity=8)
    bridge.store("hello world")
    retrieved, scores = bridge.retrieve("hello", top_k=1)
    assert retrieved.shape[0] == 1
    assert scores.shape[0] == 1
