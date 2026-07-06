"""Cross-platform corpus download utilities.

Avoids shell commands so that notebooks run reliably on Kaggle, Windows, and
Linux.
"""
import io
import zipfile
from pathlib import Path

import requests


def download_wikitext2(dest_dir="./data/wikitext-2", timeout=60):
    """Download and extract WikiText-2 (raw) using only Python stdlib + requests.

    Parameters
    ----------
    dest_dir : str or Path
        Directory where the corpus will be extracted.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    Path or None
        Path to the extracted corpus root (containing wiki.train.raw /
        wiki.valid.raw / wiki.test.raw) on success, or None on failure.
    """
    # The original Salesforce S3 bucket now returns a 301/403 and is effectively
    # unreachable. Use the verified Hugging Face mirror used by llama.cpp CI.
    url = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(dest)
        # WikiText-2 raw zip extracts to dest/wikitext-2-raw/.
        corpus_root = dest / "wikitext-2-raw"
        required = ["wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"]
        if all((corpus_root / name).exists() for name in required):
            return corpus_root
        # Fallback: search immediate subdirectories.
        for sub in dest.iterdir():
            if sub.is_dir() and all((sub / name).exists() for name in required):
                return sub
        return None
    except Exception:  # noqa: BLE001
        return None
