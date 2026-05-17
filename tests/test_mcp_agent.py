"""Behavioral tests for ``livekit_mcp_agent.py``.

These tests exercise only the ``Assistant`` class and its local tool — they do
**not** require the external Airbnb MCP server (``localhost:8089``) to be
running. The tests bypass the agent's production entrypoint and instantiate
``AgentSession`` directly without the ``mcp_servers`` list, so no network
connection to an MCP server is attempted.

Run with:
    uv run pytest -v tests/test_mcp_agent.py
"""
from __future__ import annotations

import pytest

from livekit.agents import AgentSession

from livekit_mcp_agent import Assistant
from tests.conftest import get_test_llm


@pytest.mark.asyncio
async def test_greets_user_when_addressed() -> None:
    """Agent responds to a greeting in a friendly, helpful tone."""
    llm = get_test_llm()
    async with AgentSession(llm=llm) as session:
        await session.start(Assistant())

        result = await session.run(user_input="Hi, can you help me?")

        await (
            result.expect.contains_message(role="assistant")
            .judge(
                llm,
                intent=(
                    "Responds to the user in a friendly, helpful tone and "
                    "signals willingness to assist."
                ),
            )
        )


@pytest.mark.asyncio
async def test_get_current_date_and_time_tool() -> None:
    """The local ``get_current_date_and_time`` tool is invoked when asked for the time."""
    llm = get_test_llm()
    async with AgentSession(llm=llm) as session:
        await session.start(Assistant())

        result = await session.run(user_input="What time is it right now?")

        fnc = result.expect.next_event(type="function_call")
        assert fnc.event().item.name == "get_current_date_and_time", (
            f"Expected get_current_date_and_time; got {fnc.event().item.name!r}"
        )

        result.expect.next_event().is_function_call_output()

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="States the current date and time clearly.",
            )
        )
