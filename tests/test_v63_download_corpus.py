"""Tests for cross-platform corpus download utilities."""
import io
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from snnai.utils.download_corpus import download_wikitext2


def _make_fake_zip(root_name="wikitext-2-raw-v1"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"{root_name}/wiki.train.raw", "train text")
        z.writestr(f"{root_name}/wiki.valid.raw", "valid text")
        z.writestr(f"{root_name}/wiki.test.raw", "test text")
    buf.seek(0)
    return buf.read()


def test_download_wikitext2_returns_path_on_success(tmp_path):
    mock_response = Mock()
    mock_response.content = _make_fake_zip()
    mock_response.raise_for_status = Mock()

    with patch("snnai.utils.download_corpus.requests.get", return_value=mock_response):
        result = download_wikitext2(dest_dir=str(tmp_path), timeout=1)

    assert result is not None
    assert isinstance(result, Path)
    assert (result / "wiki.train.raw").exists()
    assert (result / "wiki.valid.raw").exists()
    assert (result / "wiki.test.raw").exists()


def test_download_wikitext2_returns_none_on_failure():
    with patch("snnai.utils.download_corpus.requests.get", side_effect=Exception("network")):
        result = download_wikitext2(dest_dir="./nonexistent", timeout=1)
        assert result is None
