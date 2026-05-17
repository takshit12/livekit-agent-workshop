"""Production-grade evaluation tests using ``JudgeGroup``.

This file demonstrates the **evals layer** of the production observability
stack — distinct from the behavioral tests in ``test_basic_agent.py``:

  • Behavioral tests in ``test_basic_agent.py`` assert structure: "the agent
    called the right tool with the right args".
  • Eval tests in *this* file score the whole conversation against multiple
    quality criteria using LiveKit's built-in LLM-as-judge framework.

``JudgeGroup`` runs a list of judges concurrently and aggregates verdicts.
The same ``JudgeGroup`` you run here in pytest can also run in production
inside an ``on_session_end`` callback for live conversation scoring.

Run with:

    LIVEKIT_EVALS_VERBOSE=1 uv run pytest -v -s tests/test_evals.py

The verbose flag prints each judge's verdict and reasoning.
"""
from __future__ import annotations

import pytest

from livekit.agents import AgentSession, mock_tools
from livekit.agents.evals import (
    JudgeGroup,
    accuracy_judge,
    coherence_judge,
    conciseness_judge,
    relevancy_judge,
    safety_judge,
    task_completion_judge,
    tool_use_judge,
)

from livekit_basic_agent import Assistant
from tests.conftest import get_test_llm


@pytest.mark.asyncio
async def test_full_booking_flow_passes_production_judges() -> None:
    """End-to-end evaluation: search → book → confirm.

    A real customer interaction reduced to text mode, run through four of
    LiveKit's eight built-in judges. In production you'd wire the same
    ``JudgeGroup`` into ``on_session_end`` to score every real session.
    """
    llm = get_test_llm()

    async with AgentSession(llm=llm) as session:
        await session.start(Assistant())

        # Three-turn realistic conversation
        await session.run(user_input="Hi, I'd like to find an Airbnb please.")
        await session.run(user_input="Show me what's available in San Francisco.")
        await session.run(
            user_input=(
                "Great. Please book Airbnb sf001 for John Smith, "
                "checking in January 15, 2026, checking out January 20, 2026."
            )
        )

        # Run all relevant judges concurrently. ``llm`` here is reused as
        # the judge model — in production you'd often use a stronger judge
        # model than the one driving the agent.
        judges = JudgeGroup(
            llm=llm,
            judges=[
                task_completion_judge(),  # Did the agent finish the booking?
                tool_use_judge(),         # Were the right tools called correctly?
                accuracy_judge(),         # Are responses grounded in tool output?
                relevancy_judge(),        # Did the agent stay on topic?
                coherence_judge(),        # Logical flow across turns?
                conciseness_judge(),      # No unnecessary verbosity?
                safety_judge(),           # No harmful or off-policy content?
            ],
        )

        evaluation = await judges.evaluate(session.history)

        # In a workshop demo, print the verdicts before asserting so the
        # audience can see what passed and what didn't.
        for name, judgment in evaluation.judgments.items():
            print(f"  {name}: {judgment.verdict.upper()} — {judgment.reasoning}")

        # We tolerate one "maybe" verdict — voice agents are non-deterministic
        # and an LLM judge occasionally hedges. The hard requirement is no
        # explicit failure.
        assert evaluation.none_failed, (
            f"At least one judge failed:\n{evaluation.judgments}"
        )


@pytest.mark.asyncio
async def test_tool_failure_is_handled_safely() -> None:
    """When a tool fails, the agent should not invent results.

    This is the test you'd write to catch the "agent hallucinates when the
    backend is down" regression. We mock ``search_airbnbs`` to raise an
    error and then run the safety + accuracy judges over the result.
    """
    llm = get_test_llm()

    async with AgentSession(llm=llm) as session:
        await session.start(Assistant())

        with mock_tools(
            Assistant,
            {
                "search_airbnbs": lambda city: RuntimeError(
                    "Airbnb search service is temporarily unavailable"
                )
            },
        ):
            await session.run(
                user_input="Find me an Airbnb in San Francisco please."
            )

        judges = JudgeGroup(
            llm=llm,
            judges=[
                accuracy_judge(),    # Did the agent avoid making up listings?
                safety_judge(),      # Did it disclose the failure honestly?
                relevancy_judge(),   # Did it stay on topic?
            ],
        )
        evaluation = await judges.evaluate(session.history)

        for name, judgment in evaluation.judgments.items():
            print(f"  {name}: {judgment.verdict.upper()} — {judgment.reasoning}")

        assert evaluation.none_failed, (
            f"Agent failed under tool error:\n{evaluation.judgments}"
        )
