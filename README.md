# Althea

A voice-activated AI desktop assistant for Linux.

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Copy and fill in your API keys
cp .env.example .env
```

## Running

```bash
uv run python althea.py
```

Stop with **Ctrl+C**.

## Development

```bash
# Run tests
uv run pytest

# Run a single test file
uv run pytest tests/test_infrastructure.py -v
```
