# Contributing to mini-ir

Thank you for your interest in contributing to **mini-ir**!

The goal of this project is to provide a clean, educational, and production-principled Information Retrieval engine using **pure Python standard library** (zero external dependencies).

---

## Guiding Principles

1. **Zero External Dependencies**: All core functionality must use only Python's standard library (`math`, `re`, `collections`, `typing`, etc.). No `numpy`, `scipy`, `sklearn`, or external ML frameworks.
2. **Readability & Education First**: The codebase is designed to be read and understood by engineers, researchers, and students learning search mechanics. Code should be clean, well-commented, and directly reflect the underlying mathematical formulas.
3. **Comprehensive Tests**: Any new feature or bug fix must include tests written with Python's built-in `unittest` module.

---

## Development Setup

Clone the repository and verify the test suite:

```bash
git clone https://github.com/amitpuri/mini-ir.git
cd mini-ir

# Run test suite
python -m unittest discover -s tests -p "test_*.py" -v
```

To install in editable mode:
```bash
pip install -e .
```

---

## Running the Demo and CLI

Test the interactive demo:
```bash
python mini_ir.py
```

Test the command line interface:
```bash
python mini_ir.py --query "information retrieval" --method rrf
```

---

## Submitting Pull Requests

1. Fork the repository and create your branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes and ensure all tests pass:
   ```bash
   python -m unittest discover -s tests -p "test_*.py"
   ```
3. Commit your changes with clear, descriptive commit messages.
4. Push to your fork and submit a Pull Request against `main`.

Thank you for helping make `mini-ir` better!
