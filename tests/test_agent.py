"""Tests for the Althea Agent core (issue #5).

All ADK Runner, session, and LLM calls are mocked so these tests run without
a real Gemini API key, network connection, or downloaded model.
"""

from __future__ import annotations

from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import google.genai.types as genai_types
import pytest
from google.adk import Event
from google.adk.tools import FunctionTool

from althea.agent import AltheaAgent, _SYSTEM_PROMPT, _discover_tools
from althea.tools.echo import echo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_event(text: str, author: str = "althea") -> Event:
    """Build a final-response Event containing plain text."""
    return Event(
        author=author,
        content=genai_types.Content(
            role="model",
            parts=[genai_types.Part(text=text)],
        ),
    )


def _function_call_event(name: str, args: dict, author: str = "althea") -> Event:
    """Build a non-final Event representing an LLM function call."""
    return Event(
        author=author,
        content=genai_types.Content(
            role="model",
            parts=[
                genai_types.Part(
                    function_call=genai_types.FunctionCall(name=name, args=args)
                )
            ],
        ),
    )


async def _events_gen(*events: Event) -> AsyncGenerator[Event, None]:
    """Async generator that yields the given events in order."""
    for event in events:
        yield event


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_session() -> MagicMock:
    session = MagicMock()
    session.id = "test-session-id"
    return session


@pytest.fixture()
def mock_runner(fake_session: MagicMock) -> MagicMock:
    """Return a MagicMock InMemoryRunner whose session service creates fake sessions."""
    runner = MagicMock()
    runner.session_service.create_session = AsyncMock(return_value=fake_session)
    return runner


@pytest.fixture()
def wired_agent(mock_runner: MagicMock) -> AltheaAgent:
    """Return an AltheaAgent with its runner replaced by mock_runner.

    run_async returns a single 'ok' text event by default; override per-test
    when a different response sequence is needed.
    """
    agent = AltheaAgent(tools=[])
    agent._runner = mock_runner
    mock_runner.run_async = MagicMock(
        return_value=_events_gen(_text_event("ok"))
    )
    return agent


# ---------------------------------------------------------------------------
# AltheaAgent initialisation
# ---------------------------------------------------------------------------


class TestAltheaAgentInit:
    def test_agent_has_correct_name(self) -> None:
        agent = AltheaAgent(tools=[])
        assert agent._agent.name == "althea"

    def test_agent_uses_specified_model(self) -> None:
        agent = AltheaAgent(model="gemini-2.0-flash-lite", tools=[])
        assert agent._agent.model == "gemini-2.0-flash-lite"

    def test_agent_instruction_is_system_prompt(self) -> None:
        agent = AltheaAgent(tools=[])
        assert agent._agent.instruction == _SYSTEM_PROMPT

    def test_session_id_is_none_before_first_run(self) -> None:
        agent = AltheaAgent(tools=[])
        assert agent._session_id is None

    def test_custom_user_id_stored(self) -> None:
        agent = AltheaAgent(user_id="alice", tools=[])
        assert agent._user_id == "alice"

    def test_default_user_id_is_local(self) -> None:
        agent = AltheaAgent(tools=[])
        assert agent._user_id == "local"


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestAltheaAgentTools:
    def test_empty_tools_list_registers_nothing(self) -> None:
        agent = AltheaAgent(tools=[])
        assert agent._agent.tools == []

    def test_passed_function_tool_is_registered(self) -> None:
        tool = FunctionTool(echo)
        agent = AltheaAgent(tools=[tool])
        assert len(agent._agent.tools) == 1
        assert agent._agent.tools[0].name == "echo"

    def test_passed_plain_callable_is_accepted(self) -> None:
        # ADK also accepts plain callables (it wraps them internally).
        agent = AltheaAgent(tools=[echo])
        assert len(agent._agent.tools) == 1

    def test_tools_override_prevents_auto_discovery(self) -> None:
        """Passing tools=[] must skip _discover_tools entirely."""
        with patch("althea.agent._discover_tools") as mock_discover:
            AltheaAgent(tools=[])
        mock_discover.assert_not_called()

    def test_tools_none_triggers_auto_discovery(self) -> None:
        """Passing tools=None (the default) must call _discover_tools."""
        with patch("althea.agent._discover_tools", return_value=[]) as mock_discover:
            AltheaAgent(tools=None)
        mock_discover.assert_called_once()


# ---------------------------------------------------------------------------
# _discover_tools
# ---------------------------------------------------------------------------


class TestDiscoverTools:
    def test_echo_tool_is_discovered(self) -> None:
        tools = _discover_tools()
        names = [t.name for t in tools]
        assert "echo" in names

    def test_discovered_tools_are_function_tools(self) -> None:
        tools = _discover_tools()
        assert all(isinstance(t, FunctionTool) for t in tools)

    def test_service_tools_are_discovered(self) -> None:
        names = {tool.name for tool in _discover_tools()}
        assert {
            "play_spotify",
            "check_email",
            "send_email",
            "browser_navigate",
            "browser_send_message",
        } <= names

    def test_private_functions_are_excluded(self) -> None:
        """Functions starting with _ must never be wrapped."""
        tools = _discover_tools()
        names = [t.name for t in tools]
        assert not any(n.startswith("_") for n in names)

    def test_functions_without_docstring_are_excluded(self) -> None:
        """Callables with no docstring should not become Tools."""
        import althea.tools as tools_pkg

        # Temporarily inject a function without a docstring.
        def _no_doc() -> None:
            pass  # no docstring

        # Give it a module origin matching the package.
        import types as stdlib_types

        dummy_mod = stdlib_types.ModuleType(f"{tools_pkg.__name__}.dummy_nodoc")
        dummy_mod.__path__ = []  # type: ignore[attr-defined]
        dummy_mod.no_doc = _no_doc  # type: ignore[attr-defined]
        _no_doc.__module__ = dummy_mod.__name__
        _no_doc.__doc__ = None  # no docstring

        import sys

        sys.modules[dummy_mod.__name__] = dummy_mod

        import pkgutil

        original_iter = pkgutil.iter_modules

        def _patched_iter(path, *args, **kwargs):
            yield from original_iter(path, *args, **kwargs)
            yield pkgutil.ModuleInfo(None, "dummy_nodoc", False)  # type: ignore[arg-type]

        with patch("althea.agent.pkgutil.iter_modules", side_effect=_patched_iter):
            tools = _discover_tools()

        sys.modules.pop(dummy_mod.__name__, None)

        names = [t.name for t in tools]
        assert "no_doc" not in names


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


class TestAltheaAgentSession:
    @pytest.mark.asyncio
    async def test_session_is_created_on_first_run(
        self, wired_agent: AltheaAgent, mock_runner: MagicMock
    ) -> None:
        await wired_agent.run("hello")
        mock_runner.session_service.create_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_is_reused_on_second_run(
        self, wired_agent: AltheaAgent, mock_runner: MagicMock
    ) -> None:
        await wired_agent.run("first")
        await wired_agent.run("second")
        # session created only once
        mock_runner.session_service.create_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_id_stored_after_first_run(
        self, wired_agent: AltheaAgent
    ) -> None:
        await wired_agent.run("hello")
        assert wired_agent._session_id == "test-session-id"

    @pytest.mark.asyncio
    async def test_reset_session_starts_a_new_conversation(
        self, wired_agent: AltheaAgent, mock_runner: MagicMock
    ) -> None:
        await wired_agent.run("first conversation")
        wired_agent.reset_session()
        await wired_agent.run("new conversation")
        assert mock_runner.session_service.create_session.call_count == 2


# ---------------------------------------------------------------------------
# run() — response extraction
# ---------------------------------------------------------------------------


class TestAltheaAgentRun:
    @pytest.mark.asyncio
    async def test_run_returns_final_response_text(
        self, wired_agent: AltheaAgent, mock_runner: MagicMock
    ) -> None:
        mock_runner.run_async = MagicMock(
            return_value=_events_gen(_text_event("I heard you!"))
        )
        result = await wired_agent.run("test command")
        assert result == "I heard you!"

    @pytest.mark.asyncio
    async def test_run_ignores_non_final_events(
        self, wired_agent: AltheaAgent, mock_runner: MagicMock
    ) -> None:
        fc_event = _function_call_event("echo", {"text": "hello"})
        final_event = _text_event("Done!")
        mock_runner.run_async = MagicMock(
            return_value=_events_gen(fc_event, final_event)
        )
        result = await wired_agent.run("echo hello")
        assert result == "Done!"

    @pytest.mark.asyncio
    async def test_run_returns_empty_string_when_no_text_parts(
        self, wired_agent: AltheaAgent, mock_runner: MagicMock
    ) -> None:
        empty_event = Event(author="althea")
        mock_runner.run_async = MagicMock(return_value=_events_gen(empty_event))
        result = await wired_agent.run("silent command")
        assert result == ""

    @pytest.mark.asyncio
    async def test_run_sends_command_as_user_content(
        self, wired_agent: AltheaAgent, mock_runner: MagicMock
    ) -> None:
        await wired_agent.run("open Firefox")
        call_kwargs = mock_runner.run_async.call_args.kwargs
        message: genai_types.Content = call_kwargs["new_message"]
        assert message.role == "user"
        assert message.parts[0].text == "open Firefox"

    @pytest.mark.asyncio
    async def test_run_passes_user_id_to_runner(
        self, mock_runner: MagicMock
    ) -> None:
        agent = AltheaAgent(user_id="bob", tools=[])
        agent._runner = mock_runner
        mock_runner.run_async = MagicMock(
            return_value=_events_gen(_text_event("ok"))
        )
        await agent.run("test")
        call_kwargs = mock_runner.run_async.call_args.kwargs
        assert call_kwargs["user_id"] == "bob"

    @pytest.mark.asyncio
    async def test_run_passes_session_id_to_runner(
        self, wired_agent: AltheaAgent, mock_runner: MagicMock, fake_session: MagicMock
    ) -> None:
        await wired_agent.run("test")
        call_kwargs = mock_runner.run_async.call_args.kwargs
        assert call_kwargs["session_id"] == fake_session.id

    @pytest.mark.asyncio
    async def test_run_joins_multiple_text_parts(
        self, wired_agent: AltheaAgent, mock_runner: MagicMock
    ) -> None:
        multi_part_event = Event(
            author="althea",
            content=genai_types.Content(
                role="model",
                parts=[
                    genai_types.Part(text="Hello"),
                    genai_types.Part(text="world"),
                ],
            ),
        )
        mock_runner.run_async = MagicMock(
            return_value=_events_gen(multi_part_event)
        )
        result = await wired_agent.run("test")
        assert result == "Hello world"


# ---------------------------------------------------------------------------
# echo Tool (unit)
# ---------------------------------------------------------------------------


class TestEchoTool:
    def test_echo_returns_input_unchanged(self) -> None:
        assert echo("hello") == "hello"

    def test_echo_has_docstring(self) -> None:
        assert echo.__doc__ is not None and len(echo.__doc__) > 0

    def test_echo_accepts_empty_string(self) -> None:
        assert echo("") == ""

    def test_echo_preserves_whitespace(self) -> None:
        assert echo("  hello  ") == "  hello  "

    def test_echo_wrapped_as_function_tool(self) -> None:
        tool = FunctionTool(echo)
        assert tool.name == "echo"
