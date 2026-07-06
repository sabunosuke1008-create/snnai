"""Tests for version synchronization across project files."""
import json
from pathlib import Path

import snnai


def test_version_file_matches_package():
    version_file = Path(__file__).resolve().parents[1] / "VERSION"
    assert version_file.exists()
    file_version = version_file.read_text(encoding="utf-8").strip()
    assert snnai.__version__ == file_version


def test_kernel_metadata_title_contains_version():
    metadata_path = Path(__file__).resolve().parents[1] / "environment" / "kaggle_large_scale" / "kernel-metadata.json"
    assert metadata_path.exists()
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert any(v in data["title"] for v in ["v6-4-0", "v6.4.0", "v6-4-1", "v6.4.1", "v6-4-2", "v6.4.2", "v6-4-3", "v6.4.3"])


def test_version_is_release():
    assert snnai.__version__ == "v6.4.3"
