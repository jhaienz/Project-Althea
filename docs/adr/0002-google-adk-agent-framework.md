# Google ADK for agent orchestration

Althea uses Google's Agent Development Kit (ADK) as the agent framework rather than LangChain, raw Gemini SDK, or a custom agent loop. ADK provides native Gemini function calling, session memory, structured tool definitions, and multi-tool execution — all requirements for Althea's compound command handling.

The primary motivation is ecosystem alignment: Althea already uses Gemini as its cloud LLM, and ADK is the first-party framework for building Gemini agents. This avoids the abstraction overhead of LangChain while getting more structure than the raw SDK. The tool registration system in ADK maps directly to Althea's plugin architecture — each Tool is an ADK tool with auto-discovery from a plugins directory.

Considered: LangChain/LangGraph — rejected as over-engineered for this use case with unnecessary abstraction layers. Raw google-genai SDK — rejected because it requires hand-rolling the agent loop, memory, and multi-tool execution. Pydantic AI — a strong alternative but less integrated with Gemini's native capabilities.
