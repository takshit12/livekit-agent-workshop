"""Behavioral tests for ``livekit_basic_agent.py``.

These tests exercise the Airbnb voice agent in **text mode** — no STT/TTS,
no LiveKit room, no microphone. They use LiveKit's built-in test framework
(``session.run`` + ``result.expect``) and an LLM-based judge to validate
agent behavior.

Run with:
    uv run pytest -v tests/test_basic_agent.py

Each test costs roughly ``$0.001-0.005`` in OpenAI API usage.
"""
from __future__ import annotations

import pytest

from livekit.agents import AgentSession, mock_tools

from livekit_basic_agent import Assistant
from tests.conftest import get_test_llm


@pytest.mark.asyncio
async def test_greets_user_and_offers_help() -> None:
    """First turn: the agent greets and signals willingness to help."""
    llm = get_test_llm()
    async with AgentSession(llm=llm) as session:
        await session.start(Assistant())

        result = await session.run(user_input="Hi there!")

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent=(
                    "Greets the user in a friendly tone and signals willingness "
                    "to help. May or may not mention specific capabilities."
                ),
            )
        )


@pytest.mark.asyncio
async def test_searches_airbnbs_in_a_known_city() -> None:
    """Asking about a supported city invokes ``search_airbnbs`` with that city."""
    llm = get_test_llm()
    async with AgentSession(llm=llm) as session:
        await session.start(Assistant())

        result = await session.run(
            user_input="Can you find me some Airbnbs in San Francisco?"
        )

        fnc = result.expect.next_event().is_function_call(name="search_airbnbs")
        args = str(fnc.event().item.arguments).lower()
        assert "san francisco" in args, (
            f"Expected the city arg to mention San Francisco; got {args!r}"
        )

        result.expect.next_event().is_function_call_output()

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent=(
                    "Describes one or more Airbnb listings in San Francisco. "
                    "Mentions at least one specific property by name, address, "
                    "or price."
                ),
            )
        )


@pytest.mark.asyncio
async def test_handles_unsupported_city_gracefully() -> None:
    """An unsupported city should not produce hallucinated listings."""
    llm = get_test_llm()
    async with AgentSession(llm=llm) as session:
        await session.start(Assistant())

        result = await session.run(
            user_input="Are there any Airbnbs available in Paris?"
        )

        await (
            result.expect.contains_message(role="assistant")
            .judge(
                llm,
                intent=(
                    "Tells the user that Airbnbs in Paris are not available "
                    "in this system. Does NOT invent fake Paris listings. "
                    "May suggest the supported cities: San Francisco, "
                    "New York, or Los Angeles."
                ),
            )
        )


@pytest.mark.asyncio
async def test_books_airbnb_with_correct_parameters() -> None:
    """Two-turn flow: search first, then book — booking args must match the user's request."""
    llm = get_test_llm()
    async with AgentSession(llm=llm) as session:
        await session.start(Assistant())

        await session.run(user_input="Show me what's available in San Francisco.")

        result = await session.run(
            user_input=(
                "Great. Please book the Airbnb with ID sf001 for guest "
                "John Smith. The check-in date is January 15, 2026 and "
                "the check-out date is January 20, 2026."
            )
        )

        fnc = result.expect.next_event(type="function_call")
        assert fnc.event().item.name == "book_airbnb", (
            f"Expected book_airbnb to be called; got {fnc.event().item.name!r}"
        )

        args = str(fnc.event().item.arguments).lower()
        assert "sf001" in args, f"Expected airbnb_id sf001; got {args!r}"
        assert "john smith" in args, f"Expected guest John Smith; got {args!r}"

        result.expect.next_event().is_function_call_output()

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent=(
                    "Confirms the booking was successful. Mentions either the "
                    "confirmation number, the property name, or the guest "
                    "name as proof of the booking."
                ),
            )
        )


@pytest.mark.asyncio
async def test_searches_the_web_for_current_events() -> None:
    """Asking about current information should invoke ``search_the_web``.

    The actual DuckDuckGo call is mocked so the test:
      • doesn't depend on network connectivity in CI
      • runs deterministically (the live web changes every minute)
      • finishes in seconds
    """
    llm = get_test_llm()
    async with AgentSession(llm=llm) as session:
        await session.start(Assistant())

        with mock_tools(
            Assistant,
            {
                "search_the_web": lambda query: (
                    "Top web results:\n\n"
                    "• OpenAI releases GPT-5.4 with vision improvements\n"
                    "  OpenAI announced GPT-5.4 today with major upgrades to vision...\n"
                    "  Source: https://openai.com/blog/gpt-5-4"
                )
            },
        ):
            result = await session.run(
                user_input="What's the latest AI news this week?"
            )

            fnc = result.expect.next_event(type="function_call")
            assert fnc.event().item.name == "search_the_web", (
                f"Expected search_the_web; got {fnc.event().item.name!r}"
            )
            args = str(fnc.event().item.arguments).lower()
            assert "ai" in args or "news" in args, (
                f"Expected the query to mention AI or news; got {args!r}"
            )

            result.expect.next_event().is_function_call_output()

            await (
                result.expect.next_event()
                .is_message(role="assistant")
                .judge(
                    llm,
                    intent=(
                        "Summarizes one or more web search results about "
                        "recent AI news for the user."
                    ),
                )
            )


@pytest.mark.asyncio
async def test_get_current_date_and_time_tool() -> None:
    """Asking for the time invokes the local ``get_current_date_and_time`` tool."""
    llm = get_test_llm()
    async with AgentSession(llm=llm) as session:
        await session.start(Assistant())

        result = await session.run(user_input="What is the current date and time?")

        result.expect.next_event().is_function_call(name="get_current_date_and_time")
        result.expect.next_event().is_function_call_output()

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="States the current date and time in a natural-sounding way.",
            )
        )
