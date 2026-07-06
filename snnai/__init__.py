from pathlib import Path


def _read_version():
    here = Path(__file__).resolve().parent
    version_file = here.parent / "VERSION"
    if version_file.exists():
        return version_file.read_text(encoding="utf-8").strip()
    return "0.1.0-dev"


__version__ = _read_version()
