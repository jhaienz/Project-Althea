# Althea

A voice-activated AI desktop assistant for Linux that listens for its name, understands natural language commands, and executes tasks on the user's local machine.

## Language

### Core Concepts

**Wake Word**:
The spoken trigger phrase ("Althea") that activates the assistant from its idle listening state.
_Avoid_: Hotword, activation phrase, keyword

**Command**:
A spoken instruction from the user after the wake word is detected, transcribed and sent to the Agent for processing.
_Avoid_: Query, request, prompt

**Compound Command**:
A single utterance containing multiple intents, executed sequentially. Example: "Check my email, play some music, and open Discord."
_Avoid_: Multi-command, batch command

**Agent**:
The Gemini-powered reasoning core that interprets Commands and decides which Tools to invoke.
_Avoid_: Brain, AI, model

**Tool**:
A discrete, self-contained capability that the Agent can invoke to perform a task (e.g., launch an app, control Spotify, check email).
_Avoid_: Plugin, skill, action, function

**Overlay**:
The small floating GTK4 widget that provides visual feedback — what Althea heard, what she's doing, and her response.
_Avoid_: Widget, HUD, panel, window

### Voice Pipeline

**Transcription**:
The process of converting spoken audio to text via Faster-Whisper after the wake word is detected and speech ends.
_Avoid_: Speech recognition, STT

**Utterance**:
A continuous segment of speech bounded by silence, as detected by VAD.
_Avoid_: Audio clip, recording, speech segment

**Voice Response**:
Althea's spoken reply to the user via Piper TTS.
_Avoid_: Speech output, TTS output

### Memory & Learning

**Preference**:
A fact about the user stored in the vector database (ChromaDB) and retrieved when relevant. Examples: "Mom's name is Maria," "Likes lo-fi music."
_Avoid_: Memory, knowledge, fact

**Correction**:
Negative feedback from the user stored alongside the original command, retrieved in similar future situations to avoid repeating mistakes.
_Avoid_: Fix, override

**Fallback Mode**:
The state where Althea uses a local LLM (Ollama) instead of Gemini, triggered when the Gemini API is rate-limited or unavailable.
_Avoid_: Offline mode, degraded mode
