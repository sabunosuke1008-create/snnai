from snnai.benchmarks.code_completion import SimpleCodeCompleter


def test_code_completer_extends_snippet():
    vocab = "abcdefghijklmnopqrstuvwxyz =():\n"
    completer = SimpleCodeCompleter(vocab, hidden_size=16, time_steps=5)
    output = completer.complete("def ", max_chars=3)
    assert isinstance(output, str)
    assert len(output) == 7
