"""Knowledge distillation from a Transformer teacher to the SNN student (Phase 6.12).

The all-features SNN struggles to learn directly from hard token targets
(firing_rate collapsed to ~0 and loss went NaN in the v6.6.2 fair_compare
run). Distillation trains the student to match a *trained* Transformer
teacher's soft output distribution via a temperature-scaled KL divergence,
which provides much smoother, information-richer gradients than one-hot
targets. An optional cross-entropy term keeps the student honest on the
real targets.
"""
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from snnai.benchmarks.large_corpus_trainer import CharLMDataset, collate_fn, WarmupCosineSchedule
from snnai.benchmarks.parallel_utils import unwrap_model


def _build_loaders(text, tokenizer, seq_len, batch_size, val_ratio, drop_last=False):
    full = CharLMDataset(text, tokenizer, seq_len=seq_len)
    val_size = int(len(full) * val_ratio)
    train_size = len(full) - val_size
    if val_size == 0:
        train_ds = full
        val_ds = full
    else:
        train_ds = Subset(full, list(range(train_size)))
        val_ds = Subset(full, list(range(train_size, len(full))))

    def collate(batch):
        return collate_fn(batch, tokenizer.vocab_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, drop_last=drop_last)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate, drop_last=False)
    return train_loader, val_loader


def run_distillation(student, teacher, text, tokenizer, device='cpu', epochs=5,
                     seq_len=128, batch_size=32, time_steps=20, lr=1e-3,
                     temperature=2.0, kd_weight=0.7, ce_weight=0.3,
                     val_ratio=0.05, max_grad_norm=1.0, verbose=True,
                     drop_last=False):
    """Distill a frozen Transformer ``teacher`` into the SNN ``student``.

    Returns a history dict with train/val loss, perplexity and mean firing
    rate (the latter only populated for the student forward passes).
    """
    student = unwrap_model(student).to(device)
    teacher = unwrap_model(teacher).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(student.parameters(), lr=lr)
    total_steps = max(1, epochs * max(1, len(text) // (batch_size * seq_len)))
    scheduler = WarmupCosineSchedule(opt, warmup_steps=max(1, total_steps // 10),
                                     total_steps=max(1, total_steps), base_lr=lr, min_lr=1e-5)

    train_loader, val_loader = _build_loaders(text, tokenizer, seq_len, batch_size,
                                              val_ratio, drop_last=drop_last)

    history = {'train_loss': [], 'val_loss': [], 'train_ppl': [], 'val_ppl': [],
               'mean_firing_rate': []}

    def student_forward(one_hot):
        x = one_hot.unsqueeze(0).repeat(time_steps, 1, 1, 1).to(device)
        if hasattr(student, 'reset_memory'):
            student.reset_memory()
        out, spikes = student(x, return_spikes=True)
        return out, spikes

    for epoch in range(epochs):
        student.train()
        total_loss = 0.0
        total_tokens = 0
        total_rate = 0.0
        n_batches = 0
        for one_hot, targets in train_loader:
            one_hot = one_hot.to(device)
            targets = targets.to(device)
            opt.zero_grad()
            out, spikes = student_forward(one_hot)
            student_logits = out.reshape(-1, tokenizer.vocab_size)
            tgt = targets.reshape(-1)

            ce_loss = F.cross_entropy(student_logits, tgt, reduction='sum')
            with torch.no_grad():
                teacher_ids = one_hot.argmax(dim=-1).to(device)
                teacher_logits = teacher(teacher_ids).reshape(-1, tokenizer.vocab_size)
            kd_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction='batchmean',
            ) * (temperature ** 2)
            loss = kd_weight * kd_loss + ce_weight * ce_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
            opt.step()
            scheduler.step()

            total_loss += loss.item()
            total_tokens += tgt.numel()
            with torch.no_grad():
                layer_rates = [s.float().mean().item() for s in spikes]
                total_rate += sum(layer_rates) / max(1, len(layer_rates))
                n_batches += 1

        train_loss = total_loss / max(1, total_tokens)
        mean_rate = total_rate / max(1, n_batches)

        student.eval()
        v_loss = 0.0
        v_tokens = 0
        with torch.no_grad():
            if hasattr(student, 'reset_memory'):
                student.reset_memory()
            for one_hot, targets in val_loader:
                one_hot = one_hot.to(device)
                targets = targets.to(device)
                x = one_hot.unsqueeze(0).repeat(time_steps, 1, 1, 1).to(device)
                out, _ = student(x, return_spikes=True)
                v_loss += F.cross_entropy(out.reshape(-1, tokenizer.vocab_size),
                                          targets.reshape(-1), reduction='sum').item()
                v_tokens += targets.numel()
        val_loss = v_loss / max(1, v_tokens)

        history['train_loss'].append(train_loss)
        history['train_ppl'].append(math.exp(min(20.0, train_loss)))
        history['val_loss'].append(val_loss)
        history['val_ppl'].append(math.exp(min(20.0, val_loss)))
        history['mean_firing_rate'].append(mean_rate)

        if verbose:
            print(f'[distill] Epoch {epoch}: loss={train_loss:.4f} val_loss={val_loss:.4f} '
                  f'val_ppl={math.exp(min(20.0, val_loss)):.2f} firing_rate={mean_rate:.4f}')
    return history
