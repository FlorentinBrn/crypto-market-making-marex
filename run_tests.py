"""Harnais de test minimal sans pytest.

Ce script parcourt les fichiers test_*.py dans tests/, charge chaque
fonction qui commence par test_, exécute et rapporte le résultat. Il
fournit un `tmp_path` simple via tempfile pour les tests qui en ont besoin.
"""
from __future__ import annotations

import inspect
import sys
import tempfile
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def run_tests() -> int:
    test_files = sorted((ROOT / "tests").glob("test_*.py"))
    total = 0
    passed = 0
    failed: list[tuple[str, str]] = []

    for test_file in test_files:
        module_name = f"tests.{test_file.stem}"
        print(f"\n[MODULE] {module_name}")
        try:
            mod = __import__(module_name, fromlist=["*"])
        except Exception as e:
            print(f"  ⚠ Impossible d'importer : {e}")
            failed.append((module_name, str(e)))
            continue

        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            total += 1
            sig = inspect.signature(fn)
            kwargs = {}
            # Tests qui ont besoin de tmp_path : on fournit un Path temporaire.
            if "tmp_path" in sig.parameters:
                tmp_dir = tempfile.mkdtemp()
                kwargs["tmp_path"] = Path(tmp_dir)
            try:
                fn(**kwargs)
                print(f"  ✓ {name}")
                passed += 1
            except Exception as e:
                tb = traceback.format_exc()
                print(f"  ✗ {name}")
                print("    " + tb.replace("\n", "\n    "))
                failed.append((f"{module_name}.{name}", str(e)))

    print("\n" + "=" * 60)
    print(f"{passed}/{total} tests passés")
    if failed:
        print(f"\n{len(failed)} échec(s) :")
        for name, msg in failed:
            print(f"  - {name}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run_tests())
