# Contributing

Thank you for your interest in improving this project.

## Guidelines

- Keep changes focused and well scoped.
- Prefer readability and testability over premature optimization.
- Add or update tests when modifying strategy, analytics, or risk logic.
- Document any non-trivial assumption in the relevant module or README.

## Local workflow

```bash
pip install -e .
pytest -q
python -m crypto_mm.clean
```

## Pull requests

A good pull request should explain:

- what problem is being solved,
- why the change is useful,
- whether behavior changes are expected,
- and how the modification was tested.
