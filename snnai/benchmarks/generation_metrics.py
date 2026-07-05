"""Generation quality metrics for language models."""
import math
import torch
import torch.nn.functional as F


def perplexity(logits, targets, ignore_index=-100):
    """Compute perplexity from logits and target ids.

    Parameters
    ----------
    logits : torch.Tensor
        (batch, seq_len, vocab_size).
    targets : torch.Tensor
        (batch, seq_len) of token ids.
    ignore_index : int
        Token id to ignore in loss calculation.

    Returns
    -------
    float
        Perplexity.
    """
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                           ignore_index=ignore_index, reduction='mean')
    return math.exp(loss.item())


def token_accuracy(logits, targets, ignore_index=-100):
    """Compute next-token accuracy.

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    pred = logits.argmax(dim=-1)
    mask = targets != ignore_index
    correct = (pred == targets).masked_select(mask).sum().item()
    total = mask.sum().item()
    return correct / max(1, total)


def sequence_accuracy(logits, targets, ignore_index=-100):
    """Compute full-sequence accuracy.

    Returns
    -------
    float
        Fraction of sequences where every token is correctly predicted.
    """
    pred = logits.argmax(dim=-1)
    mask = targets != ignore_index
    matches = (pred == targets) | ~mask
    seq_correct = matches.all(dim=-1).sum().item()
    return seq_correct / max(1, targets.size(0))


def bleu_1(reference, hypothesis):
    """Compute BLEU-1 (unigram precision) for two strings.

    Parameters
    ----------
    reference : str
    hypothesis : str

    Returns
    -------
    float
    """
    ref_tokens = list(reference)
    hyp_tokens = list(hypothesis)
    if not hyp_tokens:
        return 0.0
    matches = sum(1 for t in hyp_tokens if t in ref_tokens)
    return matches / len(hyp_tokens)


def char_error_rate(reference, hypothesis):
    """Compute character-level edit distance based error rate.

    Returns
    -------
    float
        CER in [0, 1].
    """
    ref = list(reference)
    hyp = list(hypothesis)
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    # Levenshtein distance
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


def evaluate_generation(model, tokenizer, prompts, max_chars=100, device='cpu'):
    """Generate text from prompts and compute aggregate metrics.

    Parameters
    ----------
    model : torch.nn.Module
        Language model returning (batch, seq_len, vocab_size) logits.
    tokenizer : object
        Tokenizer with encode/decode.
    prompts : list[str]
        Seed strings.
    max_chars : int
        Number of characters to generate.
    device : str or torch.device

    Returns
    -------
    dict
        generated texts and aggregate metrics.
    """
    model.eval()
    from snnai.modules.language.tokenizer import SpikeEncoder
    encoder = SpikeEncoder(vocab_size=tokenizer.vocab_size, time_steps=20)
    generated = []
    with torch.no_grad():
        for prompt in prompts:
            text = prompt
            for _ in range(max_chars):
                indices = tokenizer.encode(text[-128:])
                x = torch.tensor([indices], dtype=torch.long, device=device)
                if hasattr(model, 'forward') and 'Spike' in type(model).__name__:
                    one_hot = torch.zeros(1, len(indices), tokenizer.vocab_size, device=device)
                    one_hot.scatter_(2, x.unsqueeze(2), 1.0)
                    spikes = encoder(x).to(device)
                    out = model(spikes)
                else:
                    out = model(x)
                next_id = out[:, -1, :].argmax(dim=-1).item()
                text += tokenizer.decode([next_id])
            generated.append(text)

    # Aggregate simple metrics vs prompts
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

    Parameters
    ----------
    model : torch.nn.Module
    dataloader : torch.utils.data.DataLoader
        Yields (one_hot, targets).
    tokenizer : object
    device : str or torch.device

    Returns
    -------
    dict
    """
    from snnai.modules.language.tokenizer import SpikeEncoder
    model.eval()
    encoder = SpikeEncoder(vocab_size=tokenizer.vocab_size, time_steps=20)
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    with torch.no_grad():
        for one_hot, targets in dataloader:
            one_hot = one_hot.to(device)
            targets = targets.to(device)
            x = one_hot.unsqueeze(0).repeat(20, 1, 1, 1)
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
