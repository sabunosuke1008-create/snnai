"""Generation quality metrics for language models."""
import math
from collections import Counter

import torch
import torch.nn.functional as F


def perplexity(logits, targets, ignore_index=-100):
    """Compute perplexity from logits and target ids."""
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                           ignore_index=ignore_index, reduction='mean')
    return math.exp(loss.item())


def token_accuracy(logits, targets, ignore_index=-100):
    """Compute next-token accuracy."""
    pred = logits.argmax(dim=-1)
    mask = targets != ignore_index
    correct = (pred == targets).masked_select(mask).sum().item()
    total = mask.sum().item()
    return correct / max(1, total)


def sequence_accuracy(logits, targets, ignore_index=-100):
    """Compute full-sequence accuracy."""
    pred = logits.argmax(dim=-1)
    mask = targets != ignore_index
    matches = (pred == targets) | ~mask
    seq_correct = matches.all(dim=-1).sum().item()
    return seq_correct / max(1, targets.size(0))


def bleu_1(reference, hypothesis):
    """Compute BLEU-1 (unigram precision) for two strings."""
    ref_tokens = list(reference)
    hyp_tokens = list(hypothesis)
    if not hyp_tokens:
        return 0.0
    matches = sum(1 for t in hyp_tokens if t in ref_tokens)
    return matches / len(hyp_tokens)


def char_error_rate(reference, hypothesis):
    """Compute character-level edit distance based error rate."""
    ref = list(reference)
    hyp = list(hypothesis)
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1] / len(ref)


def _encode_one_hot(indices, vocab_size, device):
    """Convert token ids to one-hot tensor."""
    one_hot = torch.zeros(indices.size(0), indices.size(1), vocab_size, device=device)
    one_hot.scatter_(2, indices.unsqueeze(2), 1.0)
    return one_hot


def _model_is_snn(model):
    """Detect SNN by presence of snntorch Leaky modules."""
    for module in model.modules():
        if module.__class__.__name__ == 'Leaky':
            return True
    return False


def _sample_next_token(logits, temperature=1.0, top_k=None):
    """Sample next token from logits with temperature and top-k filtering."""
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probs = F.softmax(logits / temperature, dim=-1)
    if top_k is not None and top_k > 0 and top_k < probs.size(-1):
        topk_probs, topk_indices = torch.topk(probs, top_k)
        sampled_idx = torch.multinomial(topk_probs, num_samples=1)
        return topk_indices.gather(-1, sampled_idx).squeeze(-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _apply_repetition_penalty(logits, recent_ids, repetition_penalty=1.0):
    """Subtract a repetition penalty from logits for recently seen tokens.

    Parameters
    ----------
    logits : torch.Tensor
        1-D logits of shape ``(vocab_size,)``.
    recent_ids : list[int]
        Recently generated token ids within the penalty window.
    repetition_penalty : float
        1.0 disables the penalty. Values > 1.0 suppress repeated tokens.

    Returns
    -------
    torch.Tensor
        Penalized logits.
    """
    if repetition_penalty <= 1.0 or not recent_ids:
        return logits
    counts = Counter(recent_ids)
    penalty = math.log(repetition_penalty)
    indices = torch.tensor(list(counts.keys()), device=logits.device, dtype=torch.long)
    # Penalty grows sub-linearly with occurrence count.
    values = torch.tensor(
        [penalty * math.log(c + 1) for c in counts.values()],
        device=logits.device, dtype=logits.dtype,
    )
    logits = logits.clone()
    logits[indices] -= values
    return logits


def _apply_token_bias(logits, bias_token_ids, bias_weight=0.0):
    """Subtract a fixed bias from logits for specified tokens.

    Useful for suppressing frequent but low-information tokens such as
    newline or space during generation.

    Parameters
    ----------
    logits : torch.Tensor
        1-D logits of shape ``(vocab_size,)``.
    bias_token_ids : list[int] or None
        Token ids to penalize.
    bias_weight : float
        Positive value subtracted from the logits of bias tokens.
        0.0 disables the bias.

    Returns
    -------
    torch.Tensor
        Biased logits.
    """
    if bias_weight <= 0.0 or not bias_token_ids:
        return logits
    logits = logits.clone()
    indices = torch.tensor(bias_token_ids, device=logits.device, dtype=torch.long)
    logits[indices] -= bias_weight
    return logits


def _sample_next_token(logits, temperature=1.0, top_k=None, top_p=None):
    """Sample next token from logits with temperature, top-k and top-p filtering.

    Parameters
    ----------
    logits : torch.Tensor
        Logits of shape ``(batch, vocab_size)`` or ``(vocab_size,)``.
    temperature : float
        Sampling temperature. Values <= 0 use greedy decoding.
    top_k : int or None
        If set, only sample from the top-k highest probability tokens.
    top_p : float or None
        Nucleus sampling threshold. If set, only sample from the smallest
        set of tokens whose cumulative probability exceeds ``top_p``.

    Returns
    -------
    torch.Tensor
        Sampled token ids of shape ``(batch,)`` or scalar.
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    if temperature <= 0:
        result = logits.argmax(dim=-1)
        return result.squeeze(0) if squeeze else result

    probs = F.softmax(logits / temperature, dim=-1)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        # Keep tokens until cumulative probability exceeds top_p.
        mask = cumsum <= top_p
        # Always keep at least one token.
        mask[:, 0] = True
        filtered_probs = sorted_probs * mask.float()
        # Renormalize over the kept tokens.
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True).clamp_min(1e-10)
        sampled_idx = torch.multinomial(filtered_probs, num_samples=1)
        result = sorted_indices.gather(-1, sampled_idx).squeeze(-1)
        return result.squeeze(0) if squeeze else result

    if top_k is not None and top_k > 0 and top_k < probs.size(-1):
        topk_probs, topk_indices = torch.topk(probs, top_k)
        sampled_idx = torch.multinomial(topk_probs, num_samples=1)
        result = topk_indices.gather(-1, sampled_idx).squeeze(-1)
        return result.squeeze(0) if squeeze else result

    result = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return result.squeeze(0) if squeeze else result


def evaluate_generation(model, tokenizer, prompts, max_chars=100, device='cpu',
                        temperature=1.0, top_k=None, top_p=None, do_sample=False,
                        repetition_penalty=1.0, penalty_window=16,
                        generation_bias_tokens=None, generation_bias_weight=0.0):
    """Generate text from prompts and compute aggregate metrics.

    Auto-detects SNN vs Transformer input format. Supports greedy or
    temperature/top-k/top-p sampling, repetition penalty, and token-level
    logit bias.
    """
    model.eval()
    from snnai.modules.language.tokenizer import SpikeEncoder
    is_snn = _model_is_snn(model)
    encoder = SpikeEncoder(vocab_size=tokenizer.vocab_size, time_steps=20)
    generated = []
    with torch.no_grad():
        for prompt in prompts:
            text = prompt
            generated_ids = []
            for _ in range(max_chars):
                indices = tokenizer.encode(text[-128:])
                x = torch.tensor([indices], dtype=torch.long, device=device)
                if is_snn:
                    spikes = encoder(x).to(device)
                    out = model(spikes)
                else:
                    out = model(x)
                next_logits = out[:, -1, :].squeeze(0)
                next_logits = _apply_repetition_penalty(
                    next_logits,
                    recent_ids=generated_ids[-penalty_window:],
                    repetition_penalty=repetition_penalty,
                )
                next_logits = _apply_token_bias(
                    next_logits,
                    bias_token_ids=generation_bias_tokens,
                    bias_weight=generation_bias_weight,
                )
                if do_sample:
                    next_id = _sample_next_token(
                        next_logits.unsqueeze(0),
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    ).item()
                else:
                    next_id = next_logits.argmax(dim=-1).item()
                generated_ids.append(next_id)
                text += tokenizer.decode([next_id])
            generated.append(text)

    bleus = [bleu_1(ref, hyp) for ref, hyp in zip(prompts, generated)]
    cers = [char_error_rate(ref, hyp) for ref, hyp in zip(prompts, generated)]
    return {
        'generated': generated,
        'bleu_1_mean': sum(bleus) / max(1, len(bleus)),
        'cer_mean': sum(cers) / max(1, len(cers)),
        'avg_length': sum(len(g) for g in generated) / max(1, len(generated)),
    }


def evaluate_model(model, dataloader, tokenizer, device='cpu'):
    """Evaluate a model on a dataloader and return perplexity/accuracy.

    Auto-detects SNN vs Transformer input format.
    """
    from snnai.modules.language.tokenizer import SpikeEncoder
    model.eval()
    is_snn = _model_is_snn(model)
    encoder = SpikeEncoder(vocab_size=tokenizer.vocab_size, time_steps=20)
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    with torch.no_grad():
        for one_hot, targets in dataloader:
            one_hot = one_hot.to(device)
            targets = targets.to(device)
            if is_snn:
                x = one_hot.unsqueeze(0).repeat(20, 1, 1, 1)
            else:
                x = one_hot.argmax(dim=-1)
            out = model(x)
            loss = F.cross_entropy(out.reshape(-1, tokenizer.vocab_size), targets.reshape(-1),
                                   reduction='sum')
            total_loss += loss.item()
            total_tokens += targets.numel()
            total_correct += (out.argmax(dim=-1) == targets).sum().item()
    avg_loss = total_loss / max(1, total_tokens)
    return {
        'loss': avg_loss,
        'ppl': math.exp(avg_loss),
        'accuracy': total_correct / max(1, total_tokens),
    }
