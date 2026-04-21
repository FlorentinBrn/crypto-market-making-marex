"""Outil de nettoyage : supprime les artefacts de run (CSV, caches)."""

from __future__ import annotations
import argparse, shutil
from pathlib import Path

DEFAULT_TARGETS = [
    "data",
    ".pytest_cache",
    "__pycache__",
]


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        print(f"Removed directory: {path}")
    elif path.exists():
        path.unlink(missing_ok=True)
        print(f"Removed file: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean generated local data and cache folders."
    )
    parser.add_argument("--root", default=".", help="Project root directory.")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    for target in DEFAULT_TARGETS:
        for p in root.rglob(target):
            # Skip crypto_mm/data/ — only the top-level data/ folder should be removed
            if target == "data" and p != root / "data":
                continue
            remove_path(p)


if __name__ == "__main__":
    main()
