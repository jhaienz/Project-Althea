# Althea

A voice-activated AI desktop assistant for Linux.

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install the browser used by the Playwright Tool
uv run playwright install chromium

# Copy and fill in your API keys
cp .env.example .env
```

Create a Spotify developer app using the redirect URI in `.env`. For Gmail,
enable the Gmail API, create desktop OAuth credentials, and save the downloaded
file at `ALTHEA_GMAIL_CREDENTIALS_PATH`. The first use opens each authorization
flow; Gmail tokens are stored in GNOME Keyring and Spotify tokens in
`~/.local/share/althea/spotify-token.json`.

Browser login state persists under `~/.local/share/althea/browser-profile`.
Install and start `ydotoold` only if native Wayland input fallback is needed.

## Running

```bash
# Optional: override the bundled en_US-amy-medium voice model
export ALTHEA_PIPER_MODEL_PATH=/path/to/en_US-amy-medium.onnx

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
