import torch
from snnai.modules.crow import WorkingMemory


def _build_trial(cue_idx, probe_idx, cue_len=10, delay_len=30, probe_len=10):
    """Build a single DMS trial. cue/probe_idx 0=A, 1=B."""
    x = torch.zeros(cue_len + delay_len + probe_len, 1, 2)
    x[:cue_len, 0, cue_idx] = 1.0
    # delay is zero input
    x[cue_len + delay_len:, 0, probe_idx] = 1.0
    return x


def test_memory_maintained_during_delay():
    wm = WorkingMemory(beta=0.85, threshold=1.0)
    x = _build_trial(cue_idx=0, probe_idx=0)
    spikes = wm(x)

    # Delay period activity.
    delay_spikes = spikes[10:40, 0, :]
    assert delay_spikes[:, 0].sum() > delay_spikes[:, 1].sum(), (
        "Memory A should remain more active than B during delay"
    )


def test_match_non_match_classification():
    wm = WorkingMemory(beta=0.85, threshold=1.0)

    # Match trial: cue A, probe A
    x_match = _build_trial(cue_idx=0, probe_idx=0)
    spikes_match = wm(x_match)
    delay_match = spikes_match[10:40]
    probe_match = torch.tensor([[1.0, 0.0]])
    scores_match = wm.classify_probe(delay_match, probe_match)
    assert scores_match[0, 0] > scores_match[0, 1], "Match trial should score match higher"

    # Non-match trial: cue A, probe B
    x_non = _build_trial(cue_idx=0, probe_idx=1)
    spikes_non = wm(x_non)
    delay_non = spikes_non[10:40]
    probe_non = torch.tensor([[0.0, 1.0]])
    scores_non = wm.classify_probe(delay_non, probe_non)
    assert scores_non[0, 0] < scores_non[0, 1], "Non-match trial should score non-match higher"
