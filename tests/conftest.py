"""Shared test configuration.

Tests need an LLM for two roles:
  1. Driving the agent under test (the ``AgentSession.llm``).
  2. Acting as judge for natural-language assertions (``.judge(llm, intent=...)``).

We use a single LLM instance for both. Two credential paths are supported:

  - **LiveKit Inference** (recommended for the workshop) — set ``LIVEKIT_URL``,
    ``LIVEKIT_API_KEY``, ``LIVEKIT_API_SECRET``. No OpenAI/Deepgram accounts
    required.
  - **OpenAI direct** — set ``OPENAI_API_KEY``. Useful if you already have
    OpenAI credentials and don't want to sign up for LiveKit Cloud.

If neither path is available, all tests are skipped with a clear reason
rather than failing with an opaque auth error.
"""
from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv(".env")
load_dotenv(".env.local")  # written by `lk app env -w`


def _has_livekit_inference() -> bool:
    return bool(
        os.getenv("LIVEKIT_API_KEY")
        and os.getenv("LIVEKIT_API_SECRET")
        and os.getenv("LIVEKIT_URL")
    )


def _has_openai() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def get_test_llm():
    """Return an LLM instance for use as both the agent driver and the judge.

    Prefers LiveKit Inference when its credentials are present, falls back to
    the OpenAI plugin otherwise. Call this from inside tests after the
    skip-check has already filtered out runs with no credentials.
    """
    if _has_livekit_inference():
        from livekit.agents import inference

        return inference.LLM(model="openai/gpt-4.1-mini")

    from livekit.plugins import openai

    return openai.LLM(model="gpt-4.1-mini")


def pytest_collection_modifyitems(config, items):
    if _has_livekit_inference() or _has_openai():
        return

    skip = pytest.mark.skip(
        reason=(
            "No LLM credentials found. Set either:\n"
            "  - LIVEKIT_URL + LIVEKIT_API_KEY + LIVEKIT_API_SECRET "
            "(LiveKit Inference, recommended for the workshop), or\n"
            "  - OPENAI_API_KEY (direct OpenAI plugin).\n"
            "Put them in .env, .env.local, or export in your shell."
        )
    )
    for item in items:
        item.add_marker(skip)
