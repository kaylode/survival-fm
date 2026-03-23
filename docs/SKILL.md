# SKILL.md — Project Skills & Conventions

## Package Management

### Always use `uv` for Python packages
- **Never** use `pip install`, `pip freeze`, or `conda`
- **Always** use `uv` commands:

```bash
# Install a package
uv add <package>

# Install dev dependency
uv add --dev <package>

# Install all dependencies from pyproject.toml
uv sync

# Run a script in the project environment
uv run python script.py

# Run a notebook
uv run jupyter notebook

# Create virtual environment
uv venv

# Lock dependencies
uv lock

# Remove a package
uv remove <package>
```

- Project dependencies live in `pyproject.toml` (not `requirements.txt`)
- Lock file: `uv.lock` (commit this)

---

## Notebooks

- Use `uv run jupyter notebook` or `uv run jupyter lab` to launch
- Convert finished experiments to `.py` scripts for reproducibility

---

## Experiments

- All experiment results go in `results/` directory
- Log metrics to a CSV or JSON file, not just printed output
- Use `argparse` or `hydra` for experiment configuration

---

## Git

- Commit after each working experiment
- Use descriptive commit messages: `feat: add TabPFN embedding pipeline`
- Never commit dataset files (already in .gitignore)

---

## Code Style

- Follow PEP 8
- Use type hints for all function signatures
- Docstrings for all public functions
