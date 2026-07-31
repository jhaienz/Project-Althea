# Copilot Instructions for Project Althea

## Build, test, and lint commands

- Install dependencies: `uv sync`
- Run Althea locally: `uv run python althea.py`
- Run full test suite: `uv run pytest`
- Run a single test file: `uv run pytest tests/test_wake_word.py -v`
- Run a single test: `uv run pytest tests/test_agent.py::TestAltheaAgentRun::test_run_returns_final_response_text -v`

## High-level architecture

- Runtime flow is linear and event-driven: `WakeWordDetector` listens continuously, triggers `VoiceActivityDetector` for one utterance capture, then `Transcriber` converts captured audio to text, and `AltheaAgent` handles intent/tool execution and returns a response (`src/althea/main.py`).
- Audio-heavy components isolate realtime concerns:
  - Wake-word detection runs in a dedicated background thread with `sounddevice` callbacks feeding openWakeWord ONNX inference (`src/althea/wake_word.py`).
  - VAD capture uses a minimal PortAudio callback that only enqueues chunks; Silero VAD inference happens on a separate capture thread to avoid callback-thread underflows (`src/althea/vad.py`).
- Agent layer is ADK-based and tool-driven:
  - `AltheaAgent` wraps Google ADK `Agent` + `InMemoryRunner`, creates one session lazily, and reuses it across commands for conversational context (`src/althea/agent.py`).
  - Tools are auto-discovered from `althea.tools` by scanning public module-level functions that have docstrings; those functions are wrapped as ADK `FunctionTool`s.
- Entry points:
  - Main runtime script: `althea.py` (adds `src/` to `sys.path` for direct execution).
  - Package entry point: `python -m althea` / console script `althea` via `althea.__main__:main`.

## Key conventions in this repository

- Keep heavyweight model/dependency imports lazy inside methods (`_load_model`, `_load_vad_model`, `_load_model` in wake-word detector), not at module import time.
- Handle unavailable audio/model dependencies by logging explicit errors and exiting the worker path cleanly instead of crashing process-wide (`wake_word.py`, `vad.py`).
- For VAD specifically, preserve the queue-based split between callback thread and inference thread; do not run model inference inside PortAudio callbacks.
- Tool discovery contract: only public functions defined in `althea.tools.*` with non-empty docstrings are intended to be exposed to the agent as tools.
- Tests avoid hardware/network/model downloads by mocking `sounddevice`, `torch.hub`, Faster-Whisper, and ADK runner/session APIs; preserve that isolation when adding tests (`tests/test_wake_word.py`, `tests/test_vad_transcription.py`, `tests/test_agent.py`).
- Pytest is configured for `--import-mode=importlib` to prevent top-level `althea.py` from shadowing `src/althea`; keep this behavior when adjusting test configuration (`pyproject.toml`).
