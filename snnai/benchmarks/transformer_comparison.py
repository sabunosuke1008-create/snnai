"""Benchmark utilities comparing SNN LM and Transformer baselines."""
import time

import torch
import torch.nn as nn

from snnai.benchmarks.large_corpus_trainer import LargeCorpusTrainer, WarmupCosineSchedule
from snnai.benchmarks.generation_metrics import evaluate_generation, evaluate_model
from snnai.benchmarks.parallel_utils import parallelize_model, unwrap_model
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

    snn_params = sum(p.numel() for p in unwrap_model(snn_model).parameters())
    transformer_params = sum(p.numel() for p in unwrap_model(transformer_model).parameters())
    return {
        'snn_latency': snn_latency,
        'transformer_latency': transformer_latency,
        'snn_parameters': snn_params,
        'transformer_parameters': transformer_params,
    }


def build_matched_transformer(snn_model, vocab_size, snn_params=None,
                              nhead=2, num_layers=2, dim_feedforward=None):
    """Build a TransformerBaseline whose parameter count approximates the SNN.

    Phase 6.10 fairness fix: the previous baseline used a fixed small config
    (8.4M params vs 3.6M SNN) which was not a fair comparison. We invert the
    parameter count to size ``d_model`` so both models sit in the same budget.
    """
    if snn_params is None:
        snn_params = sum(p.numel() for p in unwrap_model(snn_model).parameters())

    # TransformerEncoder param estimate (ignoring embedding, which both share
    # in spirit): per layer ~ 12*d_model^2 (attn + ff with 4x ff) + d_model*ff.
    # We solve for d_model so total ~ snn_params, clamped to sane bounds.
    import math
    # crude closed form: num_layers * (12*d^2 + d*4*d) = num_layers*16*d^2
    d_model = int(math.sqrt(max(16.0, snn_params / (num_layers * 16.0))))
    d_model = max(16, min(d_model, 1024))
    if d_model % nhead != 0:
        d_model = (d_model // nhead) * nhead
    if dim_feedforward is None:
        dim_feedforward = d_model * 4
    return TransformerBaseline(vocab_size=vocab_size, d_model=d_model,
                               nhead=nhead, num_layers=num_layers,
                               dim_feedforward=dim_feedforward)


def fair_compare(text, tokenizer, snn_model, transformer_model, epochs=20, seq_len=128,
                 batch_size=32, time_steps=20, device='cpu', lr=1e-3, save_dir=None,
                 label_smoothing=0.0, penalty_tokens=None, penalty_weight=0.0,
                 gen_temperature=1.0, gen_top_k=None, gen_top_p=None,
                 gen_do_sample=False, gen_repetition_penalty=1.0,
                 gen_penalty_window=16,
                 gen_generation_bias_tokens=None,
                 gen_generation_bias_weight=0.0,
                 parallel_strategy='none',
                 seeds=(0,), match_transformer=False,
                 use_distill=False, distill_temperature=2.0,
                 distill_kd_weight=0.7, distill_ce_weight=0.3,
                 distill_epochs=None, verbose=True):
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
    penalty_tokens : list[int] or None
        Token ids whose training loss is amplified to suppress over-prediction.
    penalty_weight : float
        Training loss penalty weight for penalty_tokens.
    gen_top_p : float or None
        Nucleus sampling threshold for generation.
    gen_generation_bias_tokens : list[int] or None
        Token ids whose logits are biased during generation.
    gen_generation_bias_weight : float
        Logit bias weight applied to gen_generation_bias_tokens.
    parallel_strategy : {'none', 'dp', 'ddp'}
        Multi-GPU strategy for training. 'dp' enables DataParallel in notebooks.

    Returns
    -------
    dict
        Comparison results including loss, ppl, accuracy, latency, parameters.
    """
    # Phase 6.10 fairness fix: optionally size the Transformer baseline to the
    # same parameter budget as the SNN so the comparison is apples-to-apples.
    if match_transformer:
        snn_param_count = sum(p.numel() for p in unwrap_model(snn_model).parameters())
        transformer_model = build_matched_transformer(
            snn_model, tokenizer.vocab_size, snn_params=snn_param_count)

    # Train SNN. Phase 6.15: run over multiple seeds so results are reported as
    # mean +/- std instead of a single lucky run.
    total_steps = epochs * max(1, len(text) // (batch_size * seq_len))

    def _train_snn(seed):
        torch.manual_seed(seed)
        model = snn_model.to(device)
        if parallel_strategy == 'dp' and torch.cuda.device_count() > 1:
            model = parallelize_model(model, strategy='dp', dim=1)
        opt = torch.optim.Adam(unwrap_model(model).parameters(), lr=lr)
        scheduler = WarmupCosineSchedule(opt, warmup_steps=max(1, total_steps // 10),
                                         total_steps=max(1, total_steps), base_lr=lr, min_lr=1e-5)
        trainer = LargeCorpusTrainer(model, opt, tokenizer, device=device,
                                     val_ratio=0.05, max_grad_norm=1.0, scheduler=scheduler,
                                     split_mode='temporal', label_smoothing=label_smoothing,
                                     penalty_tokens=penalty_tokens,
                                     penalty_weight=penalty_weight)
        path = f'{save_dir}/snn_best_seed{seed}.pt' if save_dir else None
        return trainer.train(text, epochs=epochs, seq_len=seq_len, batch_size=batch_size,
                             time_steps=time_steps, save_path=path, verbose=verbose,
                             drop_last=(parallel_strategy == 'dp'))

    snn_seed_histories = [_train_snn(s) for s in seeds]
    snn_history = snn_seed_histories[0]

    # Train Transformer
    transformer_model = transformer_model.to(device)
    if parallel_strategy == 'dp' and torch.cuda.device_count() > 1:
        transformer_model = parallelize_model(transformer_model, strategy='dp', dim=0)
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
                              collate_fn=lambda b: collate_fn(b, tokenizer.vocab_size),
                              drop_last=(parallel_strategy == 'dp'))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, tokenizer.vocab_size),
                            drop_last=False)

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
                'model_state_dict': unwrap_model(transformer_model).state_dict(),
                'history': transformer_history,
            }, transformer_path)

    # Phase 6.12: optionally distill the trained Transformer teacher into the
    # SNN student. This replaces the from-scratch SNN training signal with a
    # softer, information-richer target and is the main lever for making the
    # all-features SNN trainable.
    distill_history = None
    if use_distill:
        from snnai.benchmarks.distillation import run_distillation
        de = distill_epochs if distill_epochs is not None else epochs
        distill_history = run_distillation(
            snn_model, transformer_model, text, tokenizer, device=device,
            epochs=de, seq_len=seq_len, batch_size=batch_size, time_steps=time_steps,
            lr=lr, temperature=distill_temperature, kd_weight=distill_kd_weight,
            ce_weight=distill_ce_weight, verbose=True,
            drop_last=(parallel_strategy == 'dp'))
        snn_history = distill_history

    # Latency / params
    sample_input = torch.zeros(time_steps, 1, seq_len, tokenizer.vocab_size).to(device)
    comparison = compare_models(snn_model, transformer_model, sample_input)

    # Generation metrics
    prompts = ['ROMEO:', 'JULIET:', 'The ']
    snn_gen = evaluate_generation(snn_model, tokenizer, prompts, max_chars=50, device=device,
                                  temperature=gen_temperature, top_k=gen_top_k, top_p=gen_top_p,
                                  do_sample=gen_do_sample,
                                  repetition_penalty=gen_repetition_penalty,
                                  penalty_window=gen_penalty_window,
                                  generation_bias_tokens=gen_generation_bias_tokens,
                                  generation_bias_weight=gen_generation_bias_weight)
    transformer_gen = evaluate_generation(transformer_model, tokenizer, prompts, max_chars=50,
                                          device=device, temperature=gen_temperature,
                                          top_k=gen_top_k, top_p=gen_top_p,
                                          do_sample=gen_do_sample,
                                          repetition_penalty=gen_repetition_penalty,
                                          penalty_window=gen_penalty_window,
                                          generation_bias_tokens=gen_generation_bias_tokens,
                                          generation_bias_weight=gen_generation_bias_weight)

    result = {
        'snn_history': snn_history,
        'transformer_history': transformer_history,
        **comparison,
        'snn_generation': snn_gen,
        'transformer_generation': transformer_gen,
    }
    if len(seeds) > 1:
        # Phase 6.15: aggregate validation perplexity across seeds.
        import statistics
        val_ppls = [h['val_ppl'][-1] for h in snn_seed_histories if h['val_ppl']]
        result['snn_seeds'] = snn_seed_histories
        result['snn_val_ppl_mean'] = statistics.mean(val_ppls)
        result['snn_val_ppl_std'] = statistics.pstdev(val_ppls)
    if distill_history is not None:
        result['distill_history'] = distill_history
    return result
