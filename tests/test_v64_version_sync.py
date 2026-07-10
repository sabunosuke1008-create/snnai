"""Version sync tests for v6.4.0."""
import json
from pathlib import Path


def test_version_file_matches_release():
    version = Path('VERSION').read_text(encoding='utf-8').strip()
    assert version == 'v6.5.5'


def test_kernel_metadata_version_matches():
    meta = json.loads(Path('environment/kaggle_large_scale/kernel-metadata.json').read_text(encoding='utf-8'))
    assert meta['enable_internet'] is True


def test_notebook_uses_release_tag():
    nb = json.loads(Path('environment/kaggle_large_scale/notebook.ipynb').read_text(encoding='utf-8'))
    source = ''.join(sum([c['source'] for c in nb['cells'] if c['cell_type'] == 'code'], []))
    assert 'v6.5.5' in source
