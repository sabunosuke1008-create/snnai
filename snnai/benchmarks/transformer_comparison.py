"""Benchmark utilities comparing SNN LM and Transformer baselines."""
import time

import torch
import torch.nn as nn

from snnai.benchmarks.large_corpus_trainer import LargeCorpusTrainer, WarmupCosineSchedule
from snnai.benchmarks.generation_metrics import evaluate_generation, evaluate_model
from torch.utils.data import Subset


class TransformerBaseline(torch.nn.Module):
    """Minimal Transformer decoder layer baseline for comparison."""

    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """x: (batch, seq_len) long tensor."""
        emb = self.embedding(x)
        out = self.transformer(emb)
        return self.fc(out)


def compare_models(snn_model, transformer_model, sample_input, tokenizer=None):
    """Compare SNN and Transformer on latency and parameter count."""
    snn_model.eval()
    transformer_model.eval()
    device = next(snn_model.parameters()).device

    # SNN latency
    start = time.perf_counter()
    with torch.no_grad():
        _ = snn_model(sample_input)
    snn_latency = time.perf_counter() - start

    # Transformer latency
    # sample_input for SNN is (time, batch, seq, vocab); derive token ids for transformer
    transformer_input = sample_input[-1].argmax(dim=-1).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        _ = transformer_model(transformer_input)
    transformer_latency = time.perf_counter() - start

    snn_params = sum(p.numel() for p in snn_model.parameters())
    transformer_params = sum(p.numel() for p in transformer_model.parameters())
    return {
        'snn_latency': snn_latency,
        'transformer_latency': transformer_latency,
        'snn_parameters': snn_params,
        'transformer_parameters': transformer_params,
    }


def fair_compare(text, tokenizer, snn_model, transformer_model, epochs=20, seq_len=128,
                 batch_size=32, time_steps=20, device='cpu', lr=1e-3, save_dir=None,
                 label_smoothing=0.0, gen_temperature=1.0, gen_top_k=None,
                 gen_do_sample=False, gen_repetition_penalty=1.0,
                 gen_penalty_window=16):
    """Train both models on the same data and compare metrics fairly.

    Parameters
    ----------
    text : str
        Training corpus.
    tokenizer : object
    snn_model : torch.nn.Module
    transformer_model : torch.nn.Module
    epochs : int
    seq_len : int
    batch_size : int
    time_steps : int
    device : str or torch.device
    lr : float
    save_dir : str or None
        Directory prefix for saving checkpoints.
    label_smoothing : float
        Label smoothing for cross entropy.
    gen_temperature : float
        Sampling temperature for generation.
    gen_top_k : int or None
        Top-k sampling cutoff.
    gen_do_sample : bool
        Use sampling instead of greedy decoding.
    gen_repetition_penalty : float
        Repetition penalty for generation (1.0 disables).
    gen_penalty_window : int
        Number of recent tokens considered for repetition penalty.

    Returns
    -------
    dict
        Comparison results including loss, ppl, accuracy, latency, parameters.
    """
    # Train SNN
    snn_model = snn_model.to(device)
    snn_opt = torch.optim.Adam(snn_model.parameters(), lr=lr)
    total_steps = epochs * max(1, len(text) // (batch_size * seq_len))
    snn_scheduler = WarmupCosineSchedule(snn_opt, warmup_steps=max(1, total_steps // 10),
                                         total_steps=max(1, total_steps), base_lr=lr, min_lr=1e-5)
    snn_trainer = LargeCorpusTrainer(snn_model, snn_opt, tokenizer, device=device,
                                     val_ratio=0.05, max_grad_norm=1.0, scheduler=snn_scheduler,
                                     split_mode='temporal', label_smoothing=label_smoothing)
    snn_path = f'{save_dir}/snn_best.pt' if save_dir else None
    snn_history = snn_trainer.train(text, epochs=epochs, seq_len=seq_len, batch_size=batch_size,
                                    time_steps=time_steps, save_path=snn_path)

    # Train Transformer
    transformer_model = transformer_model.to(device)
    transformer_opt = torch.optim.Adam(transformer_model.parameters(), lr=lr)
    transformer_scheduler = WarmupCosineSchedule(transformer_opt, warmup_steps=max(1, total_steps // 10),
                                                 total_steps=max(1, total_steps), base_lr=lr, min_lr=1e-5)

    from snnai.benchmarks.large_corpus_trainer import CharLMDataset, collate_fn
    from torch.utils.data import DataLoader, random_split

    full_dataset = CharLMDataset(text, tokenizer, seq_len=seq_len)
    val_size = int(len(full_dataset) * 0.05)
    train_size = len(full_dataset) - val_size
    if val_size == 0:
        train_dataset, val_dataset = full_dataset, full_dataset
    else:
        # Temporal hold-out: validate on the last portion of the corpus.
        train_dataset = Subset(full_dataset, list(range(train_size)))
        val_dataset = Subset(full_dataset, list(range(train_size, len(full_dataset))))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer.vocab_size))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, tokenizer.vocab_size))

    transformer_history = {'train_loss': [], 'val_loss': [], 'train_ppl': [], 'val_ppl': []}
    best_val_loss = float('inf')
    transformer_path = f'{save_dir}/transformer_best.pt' if save_dir else None

    for epoch in range(epochs):
        transformer_model.train()
        total_loss = 0.0
        total_tokens = 0
        for one_hot, targets in train_loader:
            inputs = one_hot.argmax(dim=-1).to(device)
            targets = targets.to(device)
            transformer_opt.zero_grad()
            out = transformer_model(inputs)
            loss = nn.functional.cross_entropy(out.reshape(-1, tokenizer.vocab_size), targets.reshape(-1),
                                               reduction='sum', label_smoothing=label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), 1.0)
            transformer_opt.step()
            transformer_scheduler.step()
            total_loss += loss.item()
            total_tokens += targets.numel()
        train_loss = total_loss / max(1, total_tokens)

        transformer_model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for one_hot, targets in val_loader:
                inputs = one_hot.argmax(dim=-1).to(device)
                targets = targets.to(device)
                out = transformer_model(inputs)
                loss = nn.functional.cross_entropy(out.reshape(-1, tokenizer.vocab_size), targets.reshape(-1),
                                                   reduction='sum', label_smoothing=label_smoothing)
                total_loss += loss.item()
                total_tokens += targets.numel()
        val_loss = total_loss / max(1, total_tokens)

        transformer_history['train_loss'].append(train_loss)
        transformer_history['train_ppl'].append(torch.exp(torch.tensor(train_loss)).item())
        transformer_history['val_loss'].append(val_loss)
        transformer_history['val_ppl'].append(torch.exp(torch.tensor(val_loss)).item())

        if transformer_path and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': transformer_model.state_dict(),
                'history': transformer_history,
            }, transformer_path)

    # Latency / params
    sample_input = torch.zeros(time_steps, 1, seq_len, tokenizer.vocab_size).to(device)
    comparison = compare_models(snn_model, transformer_model, sample_input)

    # Generation metrics
    prompts = ['ROMEO:', 'JULIET:', 'The ']
    snn_gen = evaluate_generation(snn_model, tokenizer, prompts, max_chars=50, device=device,
                                  temperature=gen_temperature, top_k=gen_top_k,
                                  do_sample=gen_do_sample,
                                  repetition_penalty=gen_repetition_penalty,
                                  penalty_window=gen_penalty_window)
    transformer_gen = evaluate_generation(transformer_model, tokenizer, prompts, max_chars=50,
                                          device=device, temperature=gen_temperature,
                                          top_k=gen_top_k, do_sample=gen_do_sample,
                                          repetition_penalty=gen_repetition_penalty,
                                          penalty_window=gen_penalty_window)

    return {
        'snn_history': snn_history,
        'transformer_history': transformer_history,
        **comparison,
        'snn_generation': snn_gen,
        'transformer_generation': transformer_gen,
    }
