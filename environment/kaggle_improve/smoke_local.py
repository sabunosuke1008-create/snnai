"""Local smoke test for ImprovedSNNLM (biological-mechanism comparison).

Trains a tiny config of each variant for 1 epoch to confirm finite loss /
no crash, and probes firing rate. Mirrors what the Kaggle comparison notebook
will do at small scale.
"""
import sys
import time
import torch

sys.path.insert(0, ".")

from snnai.modules.language.bpe_tokenizer import BPETokenizer
from snnai.modules.language.large_scale_lm import LargeScaleSNNLM
from snnai.modules.language.improved_lm import ImprovedSNNLM, count_parameters
from snnai.benchmarks.transformer_comparison import (
    TransformerBaseline, fair_compare, build_matched_transformer)
from snnai.benchmarks.large_corpus_trainer import CharLMDataset, collate_fn
from torch.utils.data import DataLoader

text = ("the quick brown fox jumps over the lazy dog. " * 400)
bpe = BPETokenizer([text], vocab_size=100, max_train_bytes=500_000)
print("vocab", bpe.vocab_size, "corpus", len(text))

D = dict(embed_dim=32, hidden_dim=32, num_layers=1, max_seq_len=64,
         use_positional_encoding=True)


def probe_firing(model, device):
    ds = CharLMDataset(text, bpe, seq_len=24)
    loader = DataLoader(ds, batch_size=8, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, bpe.vocab_size))
    model.eval()
    rates = []
    with torch.no_grad():
        for i, (one_hot, _) in enumerate(loader):
            if i >= 3:
                break
            x = one_hot.unsqueeze(0).repeat(4, 1, 1, 1).to(device)
            if hasattr(model, "reset_memory"):
                model.reset_memory()
            _, spikes = model(x, return_spikes=True)
            rates.append(sum(s.float().mean().item() for s in spikes) / len(spikes))
    return sum(rates) / max(1, len(rates))


configs = {
    "current_GRU": lambda: LargeScaleSNNLM(
        bpe.vocab_size, use_sequence_recurrent=True, use_spiking_attention=False,
        use_hippocampus_gate=False, **D),
    "improved_MB_syn_GRU": lambda: ImprovedSNNLM(
        bpe.vocab_size, use_mushroom_body=True, mushroom_expand=256, mushroom_topk=32,
        use_synfire=True, use_sequence_recurrent=True, use_fixed_reservoir=False,
        use_spiking_attention=False, use_hippocampus_gate=False, **D),
    "improved_MB_syn_res": lambda: ImprovedSNNLM(
        bpe.vocab_size, use_mushroom_body=True, mushroom_expand=256, mushroom_topk=32,
        use_synfire=True, use_sequence_recurrent=False, use_fixed_reservoir=True,
        reservoir_size=128, use_spiking_attention=False, use_hippocampus_gate=False, **D),
}

for name, builder in configs.items():
    torch.manual_seed(0)
    model = builder().to("cpu")
    params = count_parameters(model)
    dummy_tf = TransformerBaseline(vocab_size=bpe.vocab_size, d_model=16, nhead=2,
                                   num_layers=1, dim_feedforward=32)
    t0 = time.time()
    res = fair_compare(text, bpe, model, dummy_tf, epochs=1, seq_len=24,
                       batch_size=8, time_steps=4, device="cpu", seeds=(0,),
                       match_transformer=True, verbose=False)
    elapsed = time.time() - t0
    snn_vp = res["snn_history"]["val_ppl"][-1]
    tf_vp = res["transformer_history"]["val_ppl"][-1]
    fr = probe_firing(model, "cpu")
    print(f"{name:22s} snn_val_ppl={snn_vp:.3f} tf_val_ppl={tf_vp:.3f} "
          f"train_sec={elapsed:.1f} params(total={params['total']},train={params['trainable']}) "
          f"firing={fr:.4f}")
