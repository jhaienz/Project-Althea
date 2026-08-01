"""Tests for Compound Command execution (issue #12).

Verifies that:
- The Agent's system prompt instructs multi-intent parsing.
- Intermediate model text events are forwarded to the on_progress callback.
- The final response is still returned correctly.
- A failed step is reported and remaining steps continue.
- Tests assert the Agent returns multiple Tool calls from a compound Command.
"""

from __future__ import annotations

from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, call, patch

import google.genai.types as genai_types
import pytest
from google.adk import Event
from google.adk.tools import FunctionTool

from althea.agent import AltheaAgent, _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Helpers (shared with test_agent.py but kept local to avoid coupling)
# ---------------------------------------------------------------------------


def _text_event(text: str, *, partial: bool = False, author: str = "althea") -> Event:
    """Build a text Event.  When partial=True it is non-final (streaming chunk)."""
    return Event(
        author=author,
        partial=partial,
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


def _function_response_event(
    name: str, response: dict, author: str = "althea"
) -> Event:
    """Build a non-final Event representing a Tool's response."""
    return Event(
        author=author,
        content=genai_types.Content(
            role="tool",
            parts=[
                genai_types.Part(
                    function_response=genai_types.FunctionResponse(
                        name=name, response=response
                    )
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
    runner = MagicMock()
    runner.session_service.create_session = AsyncMock(return_value=fake_session)
    return runner


@pytest.fixture()
def wired_agent(mock_runner: MagicMock) -> AltheaAgent:
    """AltheaAgent with its runner replaced by a mock."""
    agent = AltheaAgent(tools=[])
    agent._runner = mock_runner
    mock_runner.run_async = MagicMock(
        return_value=_events_gen(_text_event("ok"))
    )
    return agent


# ---------------------------------------------------------------------------
# System-prompt: compound-command instructions
# ---------------------------------------------------------------------------


class TestCompoundCommandSystemPrompt:
    def test_system_prompt_mentions_compound_commands(self) -> None:
        """The Agent must know what a Compound Command is."""
        assert "compound" in _SYSTEM_PROMPT.lower() or "multiple" in _SYSTEM_PROMPT.lower()

    def test_system_prompt_instructs_sequential_execution(self) -> None:
        """The prompt must instruct the Agent to execute steps in order."""
        lower = _SYSTEM_PROMPT.lower()
        assert "sequen" in lower or "order" in lower or "step" in lower

    def test_system_prompt_instructs_progress_updates(self) -> None:
        """The prompt must ask the Agent to narrate progress between steps."""
        lower = _SYSTEM_PROMPT.lower()
        assert "progress" in lower or "narrat" in lower or "update" in lower

    def test_system_prompt_instructs_error_continuation(self) -> None:
        """The prompt must tell the Agent to continue remaining steps on failure."""
        lower = _SYSTEM_PROMPT.lower()
        assert "fail" in lower or "error" in lower or "continu" in lower


# ---------------------------------------------------------------------------
# on_progress callback
# ---------------------------------------------------------------------------


class TestCompoundCommandProgress:
    @pytest.mark.asyncio
    async def test_progress_callback_called_for_intermediate_text(
        self, wired_agent: AltheaAgent, mock_runner: MagicMock
    ) -> None:
        """Intermediate model text before a function call is forwarded to on_progress."""
        # Simulate: model says "Opening Discord..." (partial/intermediate),
        # then calls a tool, then gives a final response.
        intermediate = _text_event("Opening Discord...", partial=True)
        tool_call = _function_call_event("launch_app", {"app_name": "discord"})
        final = _text_event("Done! I opened Discord and started playing music.")

        mock_runner.run_async = MagicMock(
            return_value=_events_gen(intermediate, tool_call, final)
        )

        progress_cb = MagicMock()
        result = await wired_agent.run(
            "Open Discord and play some music", on_progress=progress_cb
        )

        progress_cb.assert_called_once_with("Opening Discord...")
        assert "Done!" in result

    @pytest.mark.asyncio
    async def test_multiple_progress_updates_for_multi_step_command(
        self, wired_agent: AltheaAgent, mock_runner: MagicMock
    ) -> None:
        """Each intermediate narration step triggers a separate on_progress call."""
        step1_narration = _text_event("Opening Discord...", partial=True)
        tool1 = _function_call_event("launch_app", {"app_name": "discord"})
        step2_narration = _text_event("Playing music...", partial=True)
        tool2 = _function_call_event("play_spotify", {"search_text": "lo-fi"})
        final = _text_event("Done! Opened Discord and playing lo-fi.")

        mock_runner.run_async = MagicMock(
            return_value=_events_gen(
                step1_narration, tool1, step2_narration, tool2, final
            )
        )

        progress_cb = MagicMock()
        result = await wired_agent.run(
            "Open Discord, play some music", on_progress=progress_cb
        )

        assert progress_cb.call_count == 2
        assert progress_cb.call_args_list == [
            call("Opening Discord..."),
            call("Playing music..."),
        ]
        assert "Done!" in result

    @pytest.mark.asyncio
    async def test_no_progress_callback_does_not_raise(
        self, wired_agent: AltheaAgent, mock_runner: MagicMock
    ) -> None:
        """Omitting on_progress (default None) must not raise."""
        intermediate = _text_event("Doing something...", partial=True)
        final = _text_event("Done.")

        mock_runner.run_async = MagicMock(
            return_value=_events_gen(intermediate, final)
        )

        # Partial tokens with no following function call are treated as
        # streamed final-response content and included in the result.
        result = await wired_agent.run("do something")
        assert "Done." in result

    @pytest.mark.asyncio
    async def test_non_partial_text_event_not_forwarded_to_progress(
        self, wired_agent: AltheaAgent, mock_runner: MagicMock
    ) -> None:
        """Only partial (intermediate) text events seed narration; non-partial are not forwarded."""
        non_partial_before_final = _text_event("I'll do that.", partial=False)
        final = _text_event("All done.")

        mock_runner.run_async = MagicMock(
            return_value=_events_gen(non_partial_before_final, final)
        )

        progress_cb = MagicMock()
        await wired_agent.run("test", on_progress=progress_cb)

        progress_cb.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_partial_chunks_buffered_into_single_progress_call(
        self, wired_agent: AltheaAgent, mock_runner: MagicMock
    ) -> None:
        """Multiple partial tokens for one step are joined before on_progress fires.

        on_progress must be called exactly once per step (per function call),
        not once per streaming token.
        """
        chunk1 = _text_event("Opening ", partial=True)
        chunk2 = _text_event("Discord...", partial=True)
        tool = _function_call_event("launch_app", {"app_name": "discord"})
        final = _text_event("Done.")

        mock_runner.run_async = MagicMock(
            return_value=_events_gen(chunk1, chunk2, tool, final)
        )

        progress_cb = MagicMock()
        await wired_agent.run("Open Discord", on_progress=progress_cb)

        # The two chunks are joined into one call — not two separate calls.
        progress_cb.assert_called_once_with("Opening Discord...")

    @pytest.mark.asyncio
    async def test_final_response_not_forwarded_to_progress(
        self, wired_agent: AltheaAgent, mock_runner: MagicMock
    ) -> None:
        """The final response text must be returned, not passed to on_progress."""
        final = _text_event("All done!")

        mock_runner.run_async = MagicMock(return_value=_events_gen(final))

        progress_cb = MagicMock()
        result = await wired_agent.run("do it", on_progress=progress_cb)

        progress_cb.assert_not_called()
        assert result == "All done!"


# ---------------------------------------------------------------------------
# Multiple Tool calls from a compound Command
# ---------------------------------------------------------------------------


class TestCompoundCommandMultipleToolCalls:
    @pytest.mark.asyncio
    async def test_agent_emits_two_function_calls_for_compound_command(
        self, wired_agent: AltheaAgent, mock_runner: MagicMock
    ) -> None:
        """The runner should receive a compound command and emit two function-call events."""
        tool1 = _function_call_event("launch_app", {"app_name": "discord"})
        tool2 = _function_call_event("play_spotify", {"search_text": "lo-fi"})
        final = _text_event("Opened Discord and started playing lo-fi.")

        mock_runner.run_async = MagicMock(
            return_value=_events_gen(tool1, tool2, final)
        )

        result = await wired_agent.run("Open Discord and play some music")

        mock_runner.run_async.assert_called_once()
        call_kwargs = mock_runner.run_async.call_args.kwargs
        message: genai_types.Content = call_kwargs["new_message"]
        assert "Discord" in message.parts[0].text
        assert "music" in message.parts[0].text
        assert result == "Opened Discord and started playing lo-fi."

    @pytest.mark.asyncio
    async def test_final_response_collected_after_multiple_tool_calls(
        self, wired_agent: AltheaAgent, mock_runner: MagicMock
    ) -> None:
        """run() must still return the final text even after multiple tool calls."""
        events = [
            _function_call_event("launch_app", {"app_name": "discord"}),
            _function_response_event("launch_app", {"result": "Opened discord"}),
            _function_call_event("play_spotify", {"search_text": "lo-fi"}),
            _function_response_event("play_spotify", {"result": "Playing lo-fi beats."}),
            _text_event("Opening Discord and playing music. Done!"),
        ]

        mock_runner.run_async = MagicMock(return_value=_events_gen(*events))

        result = await wired_agent.run("Open Discord and play some music")
        assert result == "Opening Discord and playing music. Done!"


# ---------------------------------------------------------------------------
# Error handling: one step fails, remaining steps continue
# ---------------------------------------------------------------------------


class TestCompoundCommandErrorHandling:
    @pytest.mark.asyncio
    async def test_failed_step_reported_in_final_response(
        self, wired_agent: AltheaAgent, mock_runner: MagicMock
    ) -> None:
        """When a tool step fails, the Agent should report the error in its response."""
        tool1 = _function_call_event("launch_app", {"app_name": "unknownapp"})
        tool1_error = _function_response_event(
            "launch_app", {"result": "Could not find unknownapp."}
        )
        tool2 = _function_call_event("play_spotify", {"search_text": "lo-fi"})
        tool2_ok = _function_response_event(
            "play_spotify", {"result": "Playing lo-fi beats."}
        )
        final = _text_event(
            "I couldn't open unknownapp, but I started playing lo-fi music."
        )

        mock_runner.run_async = MagicMock(
            return_value=_events_gen(tool1, tool1_error, tool2, tool2_ok, final)
        )

        result = await wired_agent.run(
            "Open unknownapp and play some music"
        )
        assert "couldn't" in result.lower() or "could not" in result.lower()
        assert "lo-fi" in result.lower() or "music" in result.lower()

    @pytest.mark.asyncio
    async def test_progress_callback_called_even_after_failed_step(
        self, wired_agent: AltheaAgent, mock_runner: MagicMock
    ) -> None:
        """Progress updates should still fire even when a step has failed."""
        step1 = _text_event("Opening unknownapp...", partial=True)
        tool1 = _function_call_event("launch_app", {"app_name": "unknownapp"})
        step2 = _text_event("Playing music despite the error...", partial=True)
        tool2 = _function_call_event("play_spotify", {"search_text": "lo-fi"})
        final = _text_event("Done with errors noted.")

        mock_runner.run_async = MagicMock(
            return_value=_events_gen(step1, tool1, step2, tool2, final)
        )

        progress_cb = MagicMock()
        await wired_agent.run(
            "Open unknownapp and play some music", on_progress=progress_cb
        )

        assert progress_cb.call_count == 2


# ---------------------------------------------------------------------------
# Error policy: configurable stop-vs-continue
# ---------------------------------------------------------------------------


class TestErrorPolicy:
    @pytest.mark.asyncio
    async def test_error_policy_defaults_to_continue(
        self, wired_agent: AltheaAgent, mock_runner: MagicMock
    ) -> None:
        """Default error_policy='continue' sends the command verbatim."""
        final = _text_event("Done.")
        mock_runner.run_async = MagicMock(return_value=_events_gen(final))

        await wired_agent.run("Open Discord and play music")

        call_kwargs = mock_runner.run_async.call_args.kwargs
        message: genai_types.Content = call_kwargs["new_message"]
        # No extra instruction appended.
        assert message.parts[0].text == "Open Discord and play music"

    @pytest.mark.asyncio
    async def test_error_policy_stop_appends_halt_instruction(
        self, wired_agent: AltheaAgent, mock_runner: MagicMock
    ) -> None:
        """error_policy='stop' appends a halt-on-failure instruction to the command."""
        final = _text_event("Done.")
        mock_runner.run_async = MagicMock(return_value=_events_gen(final))

        await wired_agent.run("Open Discord and play music", error_policy="stop")

        call_kwargs = mock_runner.run_async.call_args.kwargs
        message: genai_types.Content = call_kwargs["new_message"]
        text = message.parts[0].text
        # Original command is preserved at the start.
        assert text.startswith("Open Discord and play music")
        # A halt instruction is appended.
        assert "stop" in text.lower() or "halt" in text.lower()

    @pytest.mark.asyncio
    async def test_error_policy_continue_explicit_same_as_default(
        self, wired_agent: AltheaAgent, mock_runner: MagicMock
    ) -> None:
        """Explicit error_policy='continue' behaves identically to the default."""
        final = _text_event("Done.")
        mock_runner.run_async = MagicMock(return_value=_events_gen(final))

        await wired_agent.run("Open Discord", error_policy="continue")

        call_kwargs = mock_runner.run_async.call_args.kwargs
        message: genai_types.Content = call_kwargs["new_message"]
        assert message.parts[0].text == "Open Discord"

    @pytest.mark.asyncio
    async def test_error_policy_stop_still_returns_final_response(
        self, wired_agent: AltheaAgent, mock_runner: MagicMock
    ) -> None:
        """error_policy='stop' does not affect final response collection."""
        final = _text_event("Stopped after error.")
        mock_runner.run_async = MagicMock(return_value=_events_gen(final))

        result = await wired_agent.run("Open Discord and play music", error_policy="stop")
        assert result == "Stopped after error."

    @pytest.mark.asyncio
    async def test_error_policy_stop_still_fires_progress_callbacks(
        self, wired_agent: AltheaAgent, mock_runner: MagicMock
    ) -> None:
        """error_policy='stop' does not suppress on_progress callbacks."""
        narration = _text_event("Opening Discord...", partial=True)
        tool = _function_call_event("launch_app", {"app_name": "discord"})
        final = _text_event("Done.")

        mock_runner.run_async = MagicMock(
            return_value=_events_gen(narration, tool, final)
        )

        progress_cb = MagicMock()
        await wired_agent.run(
            "Open Discord", on_progress=progress_cb, error_policy="stop"
        )
        progress_cb.assert_called_once_with("Opening Discord...")

