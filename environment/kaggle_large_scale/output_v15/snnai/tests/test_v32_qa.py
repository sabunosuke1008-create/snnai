from snnai.benchmarks.qa import SimpleQA


def test_qa_returns_answer():
    vocab = "abcdefghijklmnopqrstuvwxyz ?"
    qa = SimpleQA(vocab, hidden_size=16, time_steps=5)
    answer = qa.answer("what", "hello world")
    assert isinstance(answer, str)
    assert len(answer) == 1
