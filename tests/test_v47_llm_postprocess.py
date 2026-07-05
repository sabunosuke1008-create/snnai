from snnai.modules.language import CharTokenizer
from snnai.modules.llm_bridge.postprocess import LLMPostProcessor


def test_llm_postprocessor():
    vocab = "abcdefghijklmnopqrstuvwxyz "
    tokenizer = CharTokenizer(vocab)
    post = LLMPostProcessor(vocab_size=tokenizer.vocab_size, feature_size=4, hidden_size=8)
    output = post.process("hello", tokenizer)
    assert isinstance(output, str)
    assert len(output) >= 5
