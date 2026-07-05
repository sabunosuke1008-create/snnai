from snnai.benchmarks.dialogue import SimpleDialogue


def test_dialogue_responds():
    vocab = "abcdefghijklmnopqrstuvwxyz ?"
    dialogue = SimpleDialogue(vocab, hidden_size=16, time_steps=5, max_response=5)
    response = dialogue.respond("hello", history="")
    assert isinstance(response, str)
    assert len(response) <= 5
