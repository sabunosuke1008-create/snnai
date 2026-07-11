"""Kaggle notebook validation CI (Phase 6.9).

Runs before ``kaggle kernels push`` to prevent the recurring
Kaggle failures (IndexError, ImportError, IndentationError, broken
``\\n``-escaping) called out in ``docs/roadmap_v65.md``.

Checks
------
1. ``nbformat.validate(nb)`` - notebook JSON is well-formed.
2. Every code cell source parses with ``ast.parse`` (single + multi-cell).
3. No literal ``"\\n"`` sequences inside cell *source* strings (they must
   already be real newlines). If found, the notebook was built with
   ``"\\n"`` escapes that Kaggle fails to interpret.
4. ``kernel-metadata.json`` is consistent with the notebook:
   - ``title`` contains the release version slug.
   - the first markdown cell title contains the same version.
   - ``code_file`` matches an actual ``.ipynb`` file.

Exits non-zero on the first category of failure so it can gate a push.
"""
import ast
import json
import sys
from pathlib import Path

import nbformat


def _read_cells(nb):
    """Yield (index, cell_type, source_string) for code/markdown cells."""
    for idx, cell in enumerate(nb.get("cells", [])):
        ctype = cell.get("cell_type")
        src = cell.get("source", [])
        if isinstance(src, list):
            src = "".join(src)
        yield idx, ctype, src


def validate_structure(nb_path):
    """nbformat validation + per-cell AST parse."""
    nb_path = Path(nb_path)
    nb = nbformat.read(str(nb_path), as_version=4)
    nbformat.validate(nb)  # raises on malformed notebook

    errors = []
    code_cells = 0
    for idx, ctype, src in _read_cells(nb):
        if ctype != "code":
            continue
        code_cells += 1
        # Jupyter magics (!, %) are not valid Python syntax; strip
        # them before parsing so legitimate code is still checked.
        lines = [
            ln for ln in src.splitlines()
            if not ln.lstrip().startswith(("!", "%"))
        ]
        body = "\n".join(lines)
        if body.strip() == "":
            continue
        try:
            ast.parse(body)
        except SyntaxError as e:
            errors.append(
                f"cell {idx} ({ctype}) failed ast.parse: {e.msg} (line {e.lineno})"
            )
    if code_cells == 0:
        errors.append("notebook has no code cells")
    return errors, code_cells


def check_newline_escapes(nb_path):
    """Detect literal ``\\n`` inside cell source lists.

    A correctly serialized notebook uses real newlines; if a cell's list
    items contain backslash-n, the source was mangled on write.
    """
    raw = Path(nb_path).read_text(encoding="utf-8")
    errors = []
    nb = json.loads(raw)
    for idx, ctype, src in _read_cells(nb):
        # Each element of the source list should be a real line. A literal
        # "\\n" as part of an element (not a real newline) is the bug.
        if isinstance(src, str) and "\\n" in src:
            errors.append(
                f"cell {idx} ({ctype}) contains literal ''\\n'' escape in source"
            )
    return errors


def check_metadata_consistency(nb_path, meta_path, expected_version_slug=None):
    """Verify kernel-metadata vs notebook title + first markdown."""
    nb_path = Path(nb_path)
    meta_path = Path(meta_path)
    errors = []

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    nb = nbformat.read(str(nb_path), as_version=4)

    code_file = meta.get("code_file")
    if code_file:
        candidate = nb_path.parent / code_file
        if not candidate.exists():
            errors.append(f"code_file '{code_file}' does not exist next to notebook")

    title = meta.get("title", "")
    if expected_version_slug and expected_version_slug not in title:
        errors.append(
            f"kernel-metadata title '{title}' missing version slug "
            f"'{expected_version_slug}'"
        )

    # First markdown cell title should mention the same version slug.
    first_md = None
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            src = cell.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            first_md = src
            break
    if expected_version_slug and first_md is not None:
        if expected_version_slug not in first_md:
            errors.append(
                f"first markdown title missing version slug "
                f"'{expected_version_slug}'"
            )
    return errors


def main():
    base = Path(__file__).resolve().parent
    nb_path = base / "notebook.ipynb"
    meta_path = base / "kernel-metadata.json"

    if not nb_path.exists():
        print(f"[FAIL] notebook not found: {nb_path}")
        sys.exit(1)
    if not meta_path.exists():
        print(f"[FAIL] kernel-metadata not found: {meta_path}")
        sys.exit(1)

    version = (base.parent.parent / "VERSION").read_text(
        encoding="utf-8"
    ).strip() if (base.parent.parent / "VERSION").exists() else None
    slug = version.replace(".", "-") if version else None

    print(f"Validating {nb_path.name} (version={version}) ...")

    struct_errs, n_code = validate_structure(nb_path)
    for e in struct_errs:
        print(f"  [STRUCTURE] {e}")
    nl_errs = check_newline_escapes(nb_path)
    for e in nl_errs:
        print(f"  [NEWLINE] {e}")
    meta_errs = check_metadata_consistency(nb_path, meta_path, slug)
    for e in meta_errs:
        print(f"  [METADATA] {e}")

    total = len(struct_errs) + len(nl_errs) + len(meta_errs)
    if total == 0:
        print(f"[OK] notebook valid: {n_code} code cells, metadata consistent")
        sys.exit(0)
    print(f"[FAIL] {total} validation error(s)")
    sys.exit(1)


if __name__ == "__main__":
    main()