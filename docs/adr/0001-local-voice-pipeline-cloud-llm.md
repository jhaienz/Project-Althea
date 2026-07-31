# Local-first voice pipeline with cloud LLM fallback

Althea's voice pipeline runs entirely locally — openWakeWord for wake word detection, Silero VAD for end-of-speech detection, Faster-Whisper (OpenVINO backend) for transcription, and Piper TTS for voice responses. Only the Agent reasoning step uses a cloud service (Gemini API via Google ADK), with Ollama as a local fallback when the API is rate-limited.

This split keeps latency low for the interactive voice loop (no round-trip for audio processing), avoids cloud costs for the high-frequency audio pipeline, and ensures basic operation continues offline. The trade-off is maintaining multiple local ML models and their dependencies, but the models are small (< 200MB each) and the Intel Arc A530M GPU via OpenVINO accelerates Faster-Whisper inference.

Considered: fully cloud-based (Whisper API + cloud TTS) — rejected due to latency, cost, and offline fragility. Fully local (local LLM) — rejected because the Intel Arc GPU has limited LLM inference support, and complex reasoning tasks (email summarization, multi-step planning) need a capable model.
