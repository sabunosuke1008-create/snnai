"""Trainer for large-corpus SNN language modeling experiments."""
import json
import math
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset

from snnai.benchmarks.homeostatic_loss import HomeostaticRegularizer
from snnai.benchmarks.parallel_utils import unwrap_model


class CharLMDataset(Dataset):
    """Character-level language modeling dataset."""

    def __init__(self, text, tokenizer, seq_len=128):
        self.tokens = tokenizer.encode(text)
        self.seq_len = seq_len
        self.vocab_size = tokenizer.vocab_size

    def __len__(self):
        return max(1, len(self.tokens) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + [0] * (self.seq_len + 1 - len(chunk))
        inputs = torch.tensor(chunk[:-1], dtype=torch.long)
        targets = torch.tensor(chunk[1:], dtype=torch.long)
        return inputs, targets


def collate_fn(batch, vocab_size):
    """Convert (inputs, targets) into one-hot spike inputs."""
    inputs = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    batch_size, seq_len = inputs.shape
    one_hot = torch.zeros(batch_size, seq_len, vocab_size)
    one_hot.scatter_(2, inputs.unsqueeze(2), 1.0)
    return one_hot, targets


class WarmupCosineSchedule:
    """Simple warmup + cosine annealing learning rate schedule."""

    def __init__(self, optimizer, warmup_steps, total_steps, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            lr = self.base_lr * self.step_count / self.warmup_steps
        else:
            progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class LargeCorpusTrainer:
    """Full-featured trainer with validation, scheduling, and checkpointing."""

    def __init__(self, model, optimizer, tokenizer, device='cpu', val_ratio=0.1,
                 max_grad_norm=1.0, scheduler=None, split_mode='temporal',
                 label_smoothing=0.0, homeostatic_weight=1e-3,
                 target_firing_rate=0.12, penalty_tokens=None,
                 penalty_weight=0.0):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.device = device
        self.val_ratio = val_ratio
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        self.split_mode = split_mode
        self.label_smoothing = label_smoothing
        self.homeostatic = HomeostaticRegularizer(
            target_firing_rate=target_firing_rate,
            homeostatic_weight=homeostatic_weight,
        )
        self.penalty_tokens = penalty_tokens or []
        self.penalty_weight = penalty_weight
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_ppl': [], 'val_ppl': [],
            'lr': [], 'mean_firing_rate': [],
        }
        self.best_val_loss = float('inf')

    def _make_loaders(self, text, seq_len=128, batch_size=32, drop_last=False):
        full_dataset = CharLMDataset(text, self.tokenizer, seq_len=seq_len)
        val_size = int(len(full_dataset) * self.val_ratio)
        train_size = len(full_dataset) - val_size
        if val_size == 0:
            train_dataset = full_dataset
            val_dataset = full_dataset
        elif self.split_mode == 'temporal':
            # Hold out the last val_size chunks as validation (preserves temporal order).
            train_indices = list(range(train_size))
            val_indices = list(range(train_size, len(full_dataset)))
            train_dataset = Subset(full_dataset, train_indices)
            val_dataset = Subset(full_dataset, val_indices)
        else:
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        def collate(batch):
            return collate_fn(batch, self.tokenizer.vocab_size)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=collate, drop_last=drop_last)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=collate, drop_last=False)
        return train_loader, val_loader

    def _run_epoch(self, loader, time_steps, mode='train'):
        # Phase 6.11: clear episodic memory at the start of every epoch so that
        # training episodes never leak into validation and never accumulate
        # across epochs (a prime suspect for the all-features NaN/Inf).
        if hasattr(self.model, 'reset_memory'):
            self.model.reset_memory()
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        total_firing_rate = 0.0
        num_spike_batches = 0
        with torch.set_grad_enabled(mode == 'train'):
            for one_hot, targets in loader:
                one_hot = one_hot.to(self.device)
                targets = targets.to(self.device)
                x = one_hot.unsqueeze(0).repeat(time_steps, 1, 1, 1)
                if mode == 'train':
                    self.optimizer.zero_grad()
                out, spikes = self.model(x, return_spikes=True)
                ce_loss = torch.nn.functional.cross_entropy(
                    out.reshape(-1, self.tokenizer.vocab_size),
                    targets.reshape(-1),
                    reduction='none',
                    label_smoothing=self.label_smoothing,
                )
                # Token-level penalty to suppress over-represented tokens
                # (e.g., newline, space) during training.
                if mode == 'train' and self.penalty_tokens and self.penalty_weight > 0.0:
                    flat_targets = targets.reshape(-1)
                    penalty_mask = torch.zeros_like(flat_targets, dtype=torch.bool)
                    for token_id in self.penalty_tokens:
                        penalty_mask = penalty_mask | (flat_targets == token_id)
                    ce_loss = ce_loss * (1.0 + self.penalty_weight * penalty_mask.float())
                ce_loss = ce_loss.sum()
                # Homeostatic regularization is only applied during training to
                # avoid contaminating validation perplexity.
                if mode == 'train':
                    homeo_loss = self.homeostatic(spikes)
                    loss = ce_loss + homeo_loss
                else:
                    loss = ce_loss
                if mode == 'train':
                    loss.backward()
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                total_loss += ce_loss.item()
                total_tokens += targets.numel()
                # Log mean firing rate across all hidden layers.
                with torch.no_grad():
                    layer_rates = [s.float().mean().item() for s in spikes]
                    total_firing_rate += sum(layer_rates) / max(1, len(layer_rates))
                    num_spike_batches += 1
        avg_loss = total_loss / max(1, total_tokens)
        ppl = math.exp(avg_loss)
        mean_firing_rate = total_firing_rate / max(1, num_spike_batches)
        return avg_loss, ppl, mean_firing_rate

    def train(self, text, epochs=20, seq_len=128, batch_size=32, time_steps=20,
              save_path=None, verbose=True, drop_last=False):
        """Train and validate, returning history dictionary."""
        train_loader, val_loader = self._make_loaders(text, seq_len=seq_len, batch_size=batch_size,
                                                      drop_last=drop_last)

        for epoch in range(epochs):
            train_loss, train_ppl, train_rate = self._run_epoch(train_loader, time_steps, mode='train')
            val_loss, val_ppl, val_rate = self._run_epoch(val_loader, time_steps, mode='eval')

            lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_ppl'].append(train_ppl)
            self.history['val_loss'].append(val_loss)
            self.history['val_ppl'].append(val_ppl)
            self.history['lr'].append(lr)
            self.history['mean_firing_rate'].append(train_rate)

            if verbose:
                print(f'Epoch {epoch}: train_loss={train_loss:.4f} train_ppl={train_ppl:.2f} '
                      f'val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} '
                      f'firing_rate={train_rate:.4f} lr={lr:.2e}')

            if save_path is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(save_path, epoch, val_loss)

        return self.history

    def save_checkpoint(self, path, epoch, val_loss):
        """Save model checkpoint with metadata.

        Saves the unwrapped state dict so checkpoints remain portable between
        single-GPU and multi-GPU runs.
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
        }, path)

    def load_checkpoint(self, path):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = min(self.history.get('val_loss', [float('inf')]))
        return checkpoint
