"""Althea Agent — Gemini-powered reasoning core built on Google ADK.

Receives a transcribed Command string, reasons about intent via Gemini, and
returns the text of the final response.  Tools are auto-discovered from the
``althea.tools`` package: every module-level callable decorated with a
docstring becomes an ADK FunctionTool available to the Agent.

Usage (programmatic)::

    from althea.agent import AltheaAgent
    agent = AltheaAgent()
    response = await agent.run("open Spotify and play lo-fi music")
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from typing import Any

import google.genai.types as genai_types
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool

import althea.tools as _tools_pkg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — defines Althea's personality
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are Althea, a warm and efficient voice-activated AI desktop assistant
running on Linux.  Your personality is friendly and conversational but you
stay focused on the task at hand.  You prefer short, natural replies and
avoid unnecessary filler.

When the user gives you a command, identify the most appropriate Tool and
call it.  If no Tool fits, reply conversationally and let the user know
you are not yet able to help with that request.
"""

# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------


def _discover_tools() -> list[FunctionTool]:
    """Scan ``althea.tools`` and wrap every public callable as a FunctionTool.

    A callable is included when it:
    - is defined in (not merely imported into) an ``althea.tools.*`` module
    - has a non-empty docstring (required by ADK for tool description)
    - does not start with an underscore
    """
    tools: list[FunctionTool] = []
    pkg_path = _tools_pkg.__path__
    pkg_name = _tools_pkg.__name__

    for module_info in pkgutil.iter_modules(pkg_path):
        module = importlib.import_module(f"{pkg_name}.{module_info.name}")
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("_"):
                continue
            if obj.__module__ != module.__name__:
                # Skip re-exported symbols from other modules.
                continue
            if not obj.__doc__:
                continue
            tools.append(FunctionTool(obj))
            logger.debug("Discovered Tool: %s (from %s)", name, module.__name__)

    return tools


# ---------------------------------------------------------------------------
# AltheaAgent
# ---------------------------------------------------------------------------

_APP_NAME = "althea"


class AltheaAgent:
    """Gemini-powered reasoning core.

    Wraps an ADK ``Agent`` with a persistent ``InMemoryRunner`` so callers
    can invoke it multiple times within the same session without reloading the
    model or reconstructing state.

    Args:
        model: Gemini model identifier.  Defaults to ``gemini-2.0-flash-lite``.
        user_id: Stable identifier for the current user session.
        tools: Override the auto-discovered Tools.  Pass an empty list to
            disable discovery (useful in unit tests).
    """

    def __init__(
        self,
        *,
        model: str = "gemini-3.1-flash-lite",
        user_id: str = "local",
        tools: list[Any] | None = None,
    ) -> None:
        resolved_tools = tools if tools is not None else _discover_tools()
        self._agent = Agent(
            name=_APP_NAME,
            model=model,
            instruction=_SYSTEM_PROMPT,
            tools=resolved_tools,
        )
        self._runner = InMemoryRunner(agent=self._agent, app_name=_APP_NAME)
        self._user_id = user_id
        self._session_id: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_session(self) -> None:
        """Start a fresh Agent session on the next Command."""
        self._session_id = None

    async def run(self, command: str) -> str:
        """Send a Command string to the Agent and return its text response.

        Creates a session on first call and reuses it for subsequent calls so
        the Agent retains conversational context within one Althea session.

        Args:
            command: Transcribed speech from the user.

        Returns:
            The Agent's final text response (possibly empty string if the
            Agent produced no text output, e.g. when it only called tools
            without a closing reply).
        """
        if self._session_id is None:
            session = await self._runner.session_service.create_session(
                app_name=_APP_NAME,
                user_id=self._user_id,
            )
            self._session_id = session.id
            logger.debug("Created ADK session: %s", self._session_id)

        message = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=command)],
        )
        response_parts: list[str] = []
        async for event in self._runner.run_async(
            user_id=self._user_id,
            session_id=self._session_id,
            new_message=message,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        response_parts.append(part.text)

        response = " ".join(response_parts).strip()
        logger.info("Agent response: %s", response)
        return response
