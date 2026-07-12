"""Generate the Kaggle *small-scale* comparison notebook (current vs improved).

Runs three SNN configurations on a tiny corpus via `fair_compare` and reports
val_ppl (SNN vs matched Transformer), wall-clock training time, parameter
counts and firing rate:

  1. current_GRU        : LargeScaleSNNLM (GRU recurrence, dense embedding)
  2. improved_MB_syn_GRU: ImprovedSNNLM (Mushroom-Body encoding + Synfire timing + GRU)
  3. improved_MB_syn_res: ImprovedSNNLM (Mushroom-Body + Synfire + fixed Reservoir)

The ImprovedSNNLM implements the biological mechanisms from the investigation
(Mushroom Body / FlyHash encoding, fixed-reservoir LSM, Synfire-chain timing).
Pushed with acc='NvidiaTeslaT4' so it runs on a T4 GPU.
"""
import json
import os

NOTEBOOK = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# SNNAI v6.7.1 comparison: current vs biologically-improved (small scale)\n",
                "\n",
                "Compares three SNN configs on a tiny corpus (TinyShakespeare, BPE vocab 512):\n",
                "- **current_GRU**: LargeScaleSNNLM (GRU recurrence, dense embedding) — the existing architecture.\n",
                "- **improved_MB_syn_GRU**: ImprovedSNNLM = Mushroom-Body (FlyHash) encoding + Synfire-chain timing + GRU.\n",
                "- **improved_MB_syn_res**: ImprovedSNNLM = Mushroom-Body + Synfire + fixed Reservoir (LSM, readout-only training).\n",
                "\n",
                "Mechanisms from the bio-investigation: Mushroom Body expand-then-sparsify encoding (exp A),\n",
                "fixed reservoir LSM (exp C, cuts BPTT), Synfire-chain timing skeleton (exp B).\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install and clone the v6.7.1 tag\n",
                "!pip install -q numpy snntorch\n",
                "!rm -rf snnai\n",
                "!git clone --depth 1 --branch v6.7.1 https://github.com/sabunosuke1008-create/snnai.git\n",
                "import sys\n",
                "sys.path.insert(0, 'snnai')\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, json, time, math, requests\n",
                "import torch\n",
                "from snnai.utils.download_corpus import download_wikitext2\n",
                "from snnai.modules.language.bpe_tokenizer import BPETokenizer\n",
                "from snnai.modules.language.large_scale_lm import LargeScaleSNNLM\n",
                "from snnai.modules.language.improved_lm import ImprovedSNNLM, count_parameters\n",
                "from snnai.benchmarks.transformer_comparison import (\n",
                "    TransformerBaseline, fair_compare, build_matched_transformer)\n",
                "from snnai.benchmarks.large_corpus_trainer import CharLMDataset, collate_fn\n",
                "from torch.utils.data import DataLoader\n",
                "\n",
                "# ---- small-scale config ----\n",
                "EMBED, HID, NLAY = 64, 64, 1\n",
                "SEQ, BATCH, TSTEPS, EPOCHS = 48, 16, 6, 3\n",
                "VOCAB = 512\n",
                "SEEDS = (0,)\n",
                "MAX_SEQ = 256\n",
                "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
                "print('device', device, 'EPOCHS', EPOCHS)\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "t0 = time.time()\n",
                "corpus = requests.get(\n",
                "    'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt').text\n",
                "print('corpus length', len(corpus))\n",
                "bpe = BPETokenizer([corpus], vocab_size=VOCAB, max_train_bytes=2_000_000)\n",
                "print('bpe vocab', bpe.vocab_size)\n",
                "\n",
                "def probe_firing_rate(model, text, tokenizer, seq_len, batch_size, time_steps, n_batches=4):\n",
                "    ds = CharLMDataset(text, tokenizer, seq_len=seq_len)\n",
                "    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,\n",
                "                        collate_fn=lambda b: collate_fn(b, tokenizer.vocab_size))\n",
                "    model.eval()\n",
                "    rates = []\n",
                "    with torch.no_grad():\n",
                "        for i, (one_hot, targets) in enumerate(loader):\n",
                "            if i >= n_batches:\n",
                "                break\n",
                "            x = one_hot.unsqueeze(0).repeat(time_steps, 1, 1, 1).to(device)\n",
                "            if hasattr(model, 'reset_memory'):\n",
                "                model.reset_memory()\n",
                "            out, spikes = model(x, return_spikes=True)\n",
                "            layer_rates = [s.float().mean().item() for s in spikes]\n",
                "            rates.append(sum(layer_rates) / max(1, len(layer_rates)))\n",
                "    return sum(rates) / max(1, len(rates))\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "configs = {\n",
                "    'current_GRU': lambda: LargeScaleSNNLM(\n",
                "        bpe.vocab_size, embed_dim=EMBED, hidden_dim=HID, num_layers=NLAY,\n",
                "        max_seq_len=MAX_SEQ, output_mode='mem_last',\n",
                "        use_sequence_recurrent=True, use_positional_encoding=True,\n",
                "        use_spiking_attention=False, use_hippocampus_gate=False),\n",
                "    'improved_MB_syn_GRU': lambda: ImprovedSNNLM(\n",
                "        bpe.vocab_size, embed_dim=EMBED, hidden_dim=HID, num_layers=NLAY,\n",
                "        max_seq_len=MAX_SEQ, output_mode='mem_last',\n",
                "        use_mushroom_body=True, mushroom_expand=1024, mushroom_topk=64,\n",
                "        use_synfire=True, use_sequence_recurrent=True, use_fixed_reservoir=False,\n",
                "        use_positional_encoding=True, use_spiking_attention=False,\n",
                "        use_hippocampus_gate=False),\n",
                "    'improved_MB_syn_res': lambda: ImprovedSNNLM(\n",
                "        bpe.vocab_size, embed_dim=EMBED, hidden_dim=HID, num_layers=NLAY,\n",
                "        max_seq_len=MAX_SEQ, output_mode='mem_last',\n",
                "        use_mushroom_body=True, mushroom_expand=1024, mushroom_topk=64,\n",
                "        use_synfire=True, use_sequence_recurrent=False, use_fixed_reservoir=True,\n",
                "        reservoir_size=256, use_positional_encoding=True,\n",
                "        use_spiking_attention=False, use_hippocampus_gate=False),\n",
                "}\n",
                "\n",
                "results = {}\n",
                "for name, builder in configs.items():\n",
                "    torch.manual_seed(0)\n",
                "    snn = builder()\n",
                "    snn_p = count_parameters(snn)\n",
                "    dummy_tf = TransformerBaseline(vocab_size=bpe.vocab_size, d_model=16,\n",
                "                                  nhead=2, num_layers=1, dim_feedforward=32)\n",
                "    t_cfg = time.time()\n",
                "    res = fair_compare(\n",
                "        corpus, bpe, snn, dummy_tf, epochs=EPOCHS, seq_len=SEQ, batch_size=BATCH,\n",
                "        time_steps=TSTEPS, device=device, seeds=SEEDS, match_transformer=True,\n",
                "        use_distill=False, save_dir='/kaggle/working', verbose=True)\n",
                "    elapsed = round(time.time() - t_cfg, 1)\n",
                "    fr = probe_firing_rate(snn, corpus, bpe, SEQ, BATCH, TSTEPS)\n",
                "    results[name] = {\n",
                "        'snn_val_ppl': res['snn_history']['val_ppl'][-1],\n",
                "        'tf_val_ppl': res['transformer_history']['val_ppl'][-1],\n",
                "        'snn_params_total': snn_p['total'],\n",
                "        'snn_params_train': snn_p['trainable'],\n",
                "        'tf_params': res['transformer_parameters'],\n",
                "        'snn_latency': res['snn_latency'],\n",
                "        'tf_latency': res['transformer_latency'],\n",
                "        'firing_rate': fr,\n",
                "        'train_sec': elapsed,\n",
                "    }\n",
                "    print('===', name, '===', json.dumps(results[name], indent=2, default=str))\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('\\n===== COMPARISON SUMMARY (small scale) =====')\n",
                "print('%-22s %10s %10s %10s %10s %10s %8s' % (\n",
                "    'config', 'snn_ppl', 'tf_ppl', 'train_s', 'params', 'train_p', 'firing'))\n",
                "for name, r in results.items():\n",
                "    print('%-22s %10.2f %10.2f %10.1f %10d %10d %8.4f' % (\n",
                "        name, r['snn_val_ppl'], r['tf_val_ppl'], r['train_sec'],\n",
                "        r['snn_params_total'], r['snn_params_train'], r['firing_rate']))\n",
                "with open('/kaggle/working/comparison.json', 'w') as f:\n",
                "    json.dump(results, f, indent=2, default=str)\n",
                "print('elapsed_total_sec', round(time.time() - t0, 1))\n"
            ],
        },
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "notebook.ipynb"), "w") as f:
    json.dump(NOTEBOOK, f, indent=2)
print("wrote notebook.ipynb")
