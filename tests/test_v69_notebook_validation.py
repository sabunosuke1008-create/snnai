"""Tests for Phase 6.9 Kaggle notebook validation CI."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "environment" / "kaggle_large_scale"
NOTEBOOK = NOTEBOOK_DIR / "notebook.ipynb"
METADATA = NOTEBOOK_DIR / "kernel-metadata.json"
VALIDATE = NOTEBOOK_DIR / "validate_notebook.py"


def _run_validate():
    return subprocess.run(
        [sys.executable, "-m", "environment.kaggle_large_scale.validate_notebook"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True,
    )


def _load_validator():
    spec = importlib.util.spec_from_file_location("vld", str(VALIDATE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_notebook_validates_clean():
    """The shipped notebook passes the CI validator (exit 0)."""
    res = _run_validate()
    assert res.returncode == 0, res.stdout + res.stderr


def test_notebook_structure_ok():
    import nbformat
    nb = nbformat.read(str(NOTEBOOK), as_version=4)
    nbformat.validate(nb)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) > 0


def test_metadata_consistent_with_notebook():
    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    all_src = "".join(
        "".join(c.get("source", [])) if isinstance(c.get("source", []), list)
        else c.get("source", "")
        for c in nb["cells"]
    )
    assert meta["title"].split()[-1].lower() in all_src.lower() or "v6" in all_src


def test_validate_detects_literal_newline_escape(tmp_path):
    """A notebook with literal backslash-n escapes inside a cell fails CI."""
    import nbformat
    nb = nbformat.v4.new_notebook()
    # literally contains backslash + n (two chars), not a real newline
    broken = "x = 1" + chr(92) + "n" + chr(92) + "ny = 2"
    bad = nbformat.v4.new_code_cell(source=[broken, "z = 3"])
    nb.cells.append(bad)
    bad_path = tmp_path / "bad.ipynb"
    nbformat.write(nb, str(bad_path))
    mod = _load_validator()
    errs = mod.check_newline_escapes(bad_path)
    assert len(errs) >= 1


def test_validate_detects_bad_code_syntax(tmp_path):
    """A notebook whose code cell fails ast.parse fails CI."""
    import nbformat
    nb = nbformat.v4.new_notebook()
    bad_src = "def f(:" + chr(10) + "    pass"
    nb.cells.append(nbformat.v4.new_code_cell(source=bad_src))
    bad_path = tmp_path / "bad.ipynb"
    nbformat.write(nb, str(bad_path))
    mod = _load_validator()
    errs, _ = mod.validate_structure(bad_path)
    assert len(errs) >= 1
