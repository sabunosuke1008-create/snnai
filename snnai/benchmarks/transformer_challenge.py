"""v6.0 Transformer-level challenge benchmark harness."""
import time

import torch


def _sample_next(model, tokenizer, text, time_steps=10, temperature=1.0):
    """Sample one next character from the provided SNN model."""
    indices = torch.tensor([tokenizer.encode(text[-50:])])
    one_hot = torch.zeros(indices.size(0), indices.size(1), tokenizer.vocab_size)
    one_hot.scatter_(2, indices.unsqueeze(2), 1.0)
    x = one_hot.unsqueeze(0).repeat(time_steps, 1, 1, 1)
    out = model(x)
    logits = out[0, -1, :] / max(temperature, 1e-6)
    probs = torch.softmax(logits, dim=0)
    next_idx = int(torch.multinomial(probs, 1).item())
    return tokenizer.idx_to_char[next_idx]


def run_challenge(model, tokenizer, prompts, max_chars=50):
    """Run a set of prompts and report simple quality/speed metrics.

    This is a concept-level harness; real Transformer-level evaluation
    requires large-scale training on Kaggle.
    """
    model.eval()
    results = []
    with torch.no_grad():
        for prompt in prompts:
            start = time.perf_counter()
            text = prompt
            for _ in range(max_chars):
                text += _sample_next(model, tokenizer, text)
            latency = time.perf_counter() - start
            results.append({
                "prompt": prompt,
                "output": text,
                "latency": latency,
                "output_length": len(text),
            })
    return results


def challenge_report(results):
    """Summarize challenge results as plain text."""
    lines = ["SNNAI v6.0 Transformer-Level Challenge Report", "=" * 45]
    total_latency = sum(r["latency"] for r in results)
    total_chars = sum(r["output_length"] for r in results)
    lines.append(f"prompts: {len(results)}")
    lines.append(f"total latency: {total_latency:.4f}s")
    lines.append(f"total output chars: {total_chars}")
    for r in results:
        lines.append(f"- prompt: {r['prompt']!r} -> {r['output']!r} ({r['latency']:.4f}s)")
    return "\n".join(lines)
