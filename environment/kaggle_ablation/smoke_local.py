import os, json, time, math
import torch
from snnai.modules.language.large_scale_lm import LargeScaleSNNLM
from snnai.benchmarks.transformer_comparison import TransformerBaseline, fair_compare, build_matched_transformer
from snnai.benchmarks.large_corpus_trainer import CharLMDataset, collate_fn
from torch.utils.data import DataLoader

VOCAB = 50
EMBED, HID, NLAY = 64, 64, 1
SEQ, BATCH, TSTEPS, EPOCHS = 48, 16, 6, 2
device = 'cpu'


class D:
    vocab_size = VOCAB

    def encode(self, t):
        return [ord(c) % VOCAB for c in t]

    def decode(self, ids):
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
        return ''.join(chr((int(i) % 95) + 32) for i in ids)


bpe = D()
corpus = 'the quick brown fox jumps over the lazy dog ' * 20


def probe(model):
    ds = CharLMDataset(corpus, bpe, seq_len=SEQ)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, bpe.vocab_size))
    model.eval()
    rates = []
    with torch.no_grad():
        for i, (oh, tg) in enumerate(loader):
            if i >= 4:
                break
            x = oh.unsqueeze(0).repeat(TSTEPS, 1, 1, 1).to(device)
            if hasattr(model, 'reset_memory'):
                model.reset_memory()
            out, sp = model(x, return_spikes=True)
            lr = [s.float().mean().item() for s in sp]
            rates.append(sum(lr) / max(1, len(lr)))
    return sum(rates) / max(1, len(rates))


patterns = [
    ('P1_baseline', dict(use_sequence_recurrent=True, use_positional_encoding=False, use_hippocampus_gate=False, use_spiking_attention=False)),
    ('P2_posenc', dict(use_sequence_recurrent=True, use_positional_encoding=True, use_hippocampus_gate=False, use_spiking_attention=False)),
    ('P3_hippocampus', dict(use_sequence_recurrent=True, use_positional_encoding=True, use_hippocampus_gate=True, use_spiking_attention=False)),
    ('P4_attention', dict(use_sequence_recurrent=True, use_positional_encoding=True, use_hippocampus_gate=False, use_spiking_attention=True)),
    ('P5_all_features', dict(use_sequence_recurrent=True, use_positional_encoding=True, use_hippocampus_gate=True, use_spiking_attention=True)),
]

for name, flags in patterns:
    torch.manual_seed(0)
    snn = LargeScaleSNNLM(bpe.vocab_size, embed_dim=EMBED, hidden_dim=HID, num_layers=NLAY,
                          output_mode='mem_last', max_seq_len=256, ssa_input='spike',
                          enable_ssa_residual=True, enable_ssa_layernorm=True, **flags)
    tf = build_matched_transformer(snn, bpe.vocab_size)
    res = fair_compare(corpus, bpe, snn, tf, epochs=EPOCHS, seq_len=SEQ, batch_size=BATCH,
                       time_steps=TSTEPS, device=device, seeds=(0,), match_transformer=True, verbose=False)
    fr = probe(snn)
    snn_p = sum(p.numel() for p in snn.parameters())
    print('%-16s snn_val_ppl=%.2f tf_val_ppl=%.2f firing=%.4f snn_params=%d tf_params=%d'
          % (name, res['snn_history']['val_ppl'][-1], res['transformer_history']['val_ppl'][-1],
             fr, snn_p, res['transformer_parameters']))
