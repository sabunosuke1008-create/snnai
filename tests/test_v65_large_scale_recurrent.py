"""Tests for v6.5 LargeScaleSNNLM improvements: embedding, recurrence, positional encoding."""
import torch

from snnai.modules.language.large_scale_lm import LargeScaleSNNLM, count_parameters


def test_embedding_index_input():
    model = LargeScaleSNNLM(vocab_size=5, embed_dim=8, hidden_dim=16, num_layers=2, dropout=0.0,
                            use_sequence_recurrent=False, use_positional_encoding=False)
    x = torch.tensor([[1, 3]])  # (batch, seq_len)
    out = model(x)
    assert out.shape == (1, 2, 5)


def test_one_hot_backward_compat():
    model = LargeScaleSNNLM(vocab_size=5, embed_dim=8, hidden_dim=16, num_layers=2, dropout=0.0,
                            use_sequence_recurrent=False, use_positional_encoding=False)
    x = torch.zeros(3, 1, 2, 5)
    x[0, 0, 0, 1] = 1.0
    out = model(x)
    assert out.shape == (1, 2, 5)


def test_sequence_recurrent_changes_output():
    torch.manual_seed(0)
    model_with = LargeScaleSNNLM(vocab_size=5, embed_dim=8, hidden_dim=16, num_layers=2, dropout=0.0,
                                 use_sequence_recurrent=True, use_positional_encoding=False)
    torch.manual_seed(0)
    model_without = LargeScaleSNNLM(vocab_size=5, embed_dim=8, hidden_dim=16, num_layers=2, dropout=0.0,
                                    use_sequence_recurrent=False, use_positional_encoding=False)
    x = torch.tensor([[1, 3, 2]])
    out_with = model_with(x)
    out_without = model_without(x)
    # Sequence recurrence should change the logits.
    assert not torch.allclose(out_with, out_without)


def test_positional_encoding_changes_output():
    torch.manual_seed(0)
    model_with = LargeScaleSNNLM(vocab_size=5, embed_dim=8, hidden_dim=16, num_layers=2, dropout=0.0,
                                 use_sequence_recurrent=False, use_positional_encoding=True)
    torch.manual_seed(0)
    model_without = LargeScaleSNNLM(vocab_size=5, embed_dim=8, hidden_dim=16, num_layers=2, dropout=0.0,
                                    use_sequence_recurrent=False, use_positional_encoding=False)
    x = torch.tensor([[1, 3, 2]])
    out_with = model_with(x)
    out_without = model_without(x)
    # Positional encoding should change the logits.
    assert not torch.allclose(out_with, out_without)


def test_sequence_recurrent_has_gru_parameters():
    model = LargeScaleSNNLM(vocab_size=5, embed_dim=8, hidden_dim=16, num_layers=1, dropout=0.0,
                            use_sequence_recurrent=True)
    assert hasattr(model, 'seq_recurrent')
    counts = count_parameters(model)
    assert counts['total'] > 0


def test_positional_encoding_has_pos_embed_parameters():
    model = LargeScaleSNNLM(vocab_size=5, embed_dim=8, hidden_dim=16, num_layers=1, dropout=0.0,
                            use_positional_encoding=True)
    assert hasattr(model, 'pos_embed')


def test_return_spikes_with_recurrent():
    model = LargeScaleSNNLM(vocab_size=5, embed_dim=8, hidden_dim=16, num_layers=2, dropout=0.0,
                            use_sequence_recurrent=True, use_positional_encoding=True)
    x = torch.tensor([[1, 3, 2]])
    logits, spikes = model(x, return_spikes=True)
    assert logits.shape == (1, 3, 5)
    assert len(spikes) == 2
    assert spikes[0].shape[1:] == (1, 3, 16)
