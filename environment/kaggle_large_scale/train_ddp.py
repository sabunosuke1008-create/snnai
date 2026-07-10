"""DDP-based multi-GPU training script for SNNAI v6.5+ LargeScaleSNNLM.

Usage in a Kaggle notebook cell after cloning the repo and downloading deps:
    !python train_ddp.py

Each spawned process owns one GPU and one model replica. This avoids the
DataParallel scatter/gather incompatibility with snntorch LIF stateful
neurons observed in v6.5.0--v6.5.3.
"""
import os
import sys
import math
import json
import argparse
import pickle

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, '/kaggle/working/snnai')
from snnai.modules.language import BPETokenizer
from snnai.modules.language.large_scale_lm import LargeScaleSNNLM
from snnai.benchmarks.large_corpus_trainer import CharLMDataset, WarmupCosineSchedule
from snnai.benchmarks.homeostatic_loss import HomeostaticRegularizer
from snnai.benchmarks.transformer_comparison import TransformerBaseline
from snnai.benchmarks.parallel_utils import unwrap_model


def _collate(batch, vocab_size):
    inputs = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    b, l = inputs.shape
    one_hot = torch.zeros(b, l, vocab_size)
    one_hot.scatter_(2, inputs.unsqueeze(2), 1.0)
    return one_hot, targets


def setup(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def download_corpus():
    import requests
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    corpus = requests.get(url).text
    try:
        from snnai.utils.download_corpus import download_wikitext2
        wt_root = download_wikitext2(dest_dir='/kaggle/working/wikitext-2', timeout=120)
        if wt_root is not None:
            wt_files = [
                os.path.join(wt_root, 'wiki.train.raw'),
                os.path.join(wt_root, 'wiki.valid.raw'),
                os.path.join(wt_root, 'wiki.test.raw'),
            ]
            wt_text = ''.join(open(f).read() for f in wt_files if os.path.exists(f))
            if wt_text:
                corpus = corpus + '\n' + wt_text
    except Exception as exc:
        print(f'WikiText-2 download failed: {exc}')
    return corpus


def train_fn(rank, world_size, args):
    setup(rank, world_size, args.master_port)

    # Data: rank 0 downloads, all ranks read from file.
    corpus_path = '/kaggle/working/corpus.txt'
    if rank == 0:
        corpus = download_corpus()
        with open(corpus_path, 'w', encoding='utf-8') as f:
            f.write(corpus)
        print(f'[rank 0] corpus length: {len(corpus)}')
    dist.barrier()
    with open(corpus_path, encoding='utf-8') as f:
        corpus = f.read()

    # Tokenizer: rank 0 trains BPE, saves; others load.
    bpe_path = '/kaggle/working/bpe.pkl'
    if rank == 0:
        bpe = BPETokenizer([corpus], vocab_size=args.vocab_size,
                           max_train_bytes=args.bpe_train_bytes)
        with open(bpe_path, 'wb') as f:
            pickle.dump(bpe, f)
        print(f'[rank 0] BPE vocab size: {bpe.vocab_size}')
    dist.barrier()
    with open(bpe_path, 'rb') as f:
        bpe = pickle.load(f)
    vocab_size = bpe.vocab_size

    # Dataset & DistributedSampler (temporal split, train on first part, val on last).
    dataset = CharLMDataset(corpus, bpe, seq_len=args.seq_len)
    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                       shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank,
                                     shuffle=False, drop_last=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              collate_fn=lambda b: _collate(b, vocab_size), drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler,
                            collate_fn=lambda b: _collate(b, vocab_size), drop_last=False)

    # Models wrapped with DDP (each rank owns one replica).
    snn_model = LargeScaleSNNLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=0.2,
        output_mode='mem_last',
        use_sequence_recurrent=True,
        use_positional_encoding=True,
        max_seq_len=args.seq_len,
    ).to(rank)
    snn_model = DDP(snn_model, device_ids=[rank])

    transformer_model = TransformerBaseline(
        vocab_size=vocab_size,
        d_model=args.tx_d_model,
        nhead=args.tx_nhead,
        num_layers=args.tx_num_layers,
        dim_feedforward=args.tx_dim_ff,
    ).to(rank)
    transformer_model = DDP(transformer_model, device_ids=[rank])

    # Optimizers & schedulers.
    total_steps = args.epochs * max(1, len(train_loader))
    snn_opt = torch.optim.AdamW(snn_model.parameters(), lr=args.lr, weight_decay=0.01)
    snn_sched = WarmupCosineSchedule(snn_opt, warmup_steps=max(1, total_steps // 10),
                                     total_steps=max(1, total_steps), base_lr=args.lr, min_lr=1e-5)
    tx_opt = torch.optim.AdamW(transformer_model.parameters(), lr=args.lr, weight_decay=0.01)
    tx_sched = WarmupCosineSchedule(tx_opt, warmup_steps=max(1, total_steps // 10),
                                    total_steps=max(1, total_steps), base_lr=args.lr, min_lr=1e-5)
    homeo = HomeostaticRegularizer(target_firing_rate=0.12, homeostatic_weight=1e-3)

    history = {
        'snn': {'train_loss': [], 'val_loss': [], 'train_ppl': [], 'val_ppl': []},
        'transformer': {'train_loss': [], 'val_loss': [], 'train_ppl': [], 'val_ppl': []},
    }

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        # SNN training epoch.
        snn_model.train()
        snn_loss_sum, snn_tokens = 0.0, 0
        for one_hot, targets in train_loader:
            one_hot, targets = one_hot.to(rank), targets.to(rank)
            x = one_hot.unsqueeze(0).repeat(args.time_steps, 1, 1, 1)
            snn_opt.zero_grad()
            out, spikes = snn_model(x, return_spikes=True)
            ce = F.cross_entropy(out.reshape(-1, vocab_size), targets.reshape(-1),
                                 label_smoothing=0.1, reduction='sum')
            homeo_loss = homeo(spikes)
            loss = ce + homeo_loss * targets.numel()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(snn_model.parameters(), 1.0)
            snn_opt.step()
            snn_sched.step()
            snn_loss_sum += ce.item()
            snn_tokens += targets.numel()
        snn_train_loss = snn_loss_sum / max(1, snn_tokens)

        # Transformer training epoch.
        transformer_model.train()
        tx_loss_sum, tx_tokens = 0.0, 0
        for one_hot, targets in train_loader:
            inputs = one_hot.argmax(dim=-1).to(rank)
            targets = targets.to(rank)
            tx_opt.zero_grad()
            out = transformer_model(inputs)
            loss = F.cross_entropy(out.reshape(-1, vocab_size), targets.reshape(-1),
                                   label_smoothing=0.1, reduction='sum')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), 1.0)
            tx_opt.step()
            tx_sched.step()
            tx_loss_sum += loss.item()
            tx_tokens += targets.numel()
        tx_train_loss = tx_loss_sum / max(1, tx_tokens)

        # Validation with all-reduce across ranks.
        snn_model.eval()
        transformer_model.eval()
        with torch.no_grad():
            snn_val_sum, snn_val_tok = 0.0, 0
            tx_val_sum, tx_val_tok = 0.0, 0
            for one_hot, targets in val_loader:
                one_hot, targets = one_hot.to(rank), targets.to(rank)
                x = one_hot.unsqueeze(0).repeat(args.time_steps, 1, 1, 1)
                out, _ = snn_model(x, return_spikes=True)
                ce = F.cross_entropy(out.reshape(-1, vocab_size), targets.reshape(-1),
                                     reduction='sum')
                snn_val_sum += ce.item()
                snn_val_tok += targets.numel()
                inputs = one_hot.argmax(dim=-1)
                out = transformer_model(inputs)
                ce = F.cross_entropy(out.reshape(-1, vocab_size), targets.reshape(-1),
                                     reduction='sum')
                tx_val_sum += ce.item()
                tx_val_tok += targets.numel()
        metrics = torch.tensor([snn_val_sum, float(snn_val_tok),
                                tx_val_sum, float(tx_val_tok)],
                               dtype=torch.float64, device=rank)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        snn_val_loss_g = metrics[0].item() / max(1, metrics[1].item())
        tx_val_loss_g = metrics[2].item() / max(1, metrics[3].item())

        if rank == 0:
            snn_train_ppl = math.exp(min(20.0, snn_train_loss))
            snn_val_ppl = math.exp(min(20.0, snn_val_loss_g))
            tx_train_ppl = math.exp(min(20.0, tx_train_loss))
            tx_val_ppl = math.exp(min(20.0, tx_val_loss_g))
            print(f'Epoch {epoch}: SNN train_loss={snn_train_loss:.4f} ppl={snn_train_ppl:.2f} '
                  f'val_ppl={snn_val_ppl:.2f} | TX train_loss={tx_train_loss:.4f} ppl={tx_train_ppl:.2f} '
                  f'val_ppl={tx_val_ppl:.2f}')
            history['snn']['train_loss'].append(snn_train_loss)
            history['snn']['val_loss'].append(snn_val_loss_g)
            history['snn']['train_ppl'].append(snn_train_ppl)
            history['snn']['val_ppl'].append(snn_val_ppl)
            history['transformer']['train_loss'].append(tx_train_loss)
            history['transformer']['val_loss'].append(tx_val_loss_g)
            history['transformer']['train_ppl'].append(tx_train_ppl)
            history['transformer']['val_ppl'].append(tx_val_ppl)

    if rank == 0:
        torch.save({
            'snn_state_dict': unwrap_model(snn_model).state_dict(),
            'tx_state_dict': unwrap_model(transformer_model).state_dict(),
            'vocab_size': vocab_size,
            'history': history,
        }, '/kaggle/working/snnai_v6_ddp.pt')
        with open('/kaggle/working/snnai_v6_ddp_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        print('Saved /kaggle/working/snnai_v6_ddp.pt')

    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--time_steps', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--val_ratio', type=float, default=0.05)
    parser.add_argument('--vocab_size', type=int, default=2048)
    parser.add_argument('--bpe_train_bytes', type=int, default=2_000_000)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--tx_d_model', type=int, default=256)
    parser.add_argument('--tx_nhead', type=int, default=4)
    parser.add_argument('--tx_num_layers', type=int, default=4)
    parser.add_argument('--tx_dim_ff', type=int, default=1024)
    parser.add_argument('--master_port', type=int, default=29501)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError('No CUDA devices available for DDP.')
    print(f'Spawning {world_size} DDP processes')
    mp.spawn(train_fn, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
