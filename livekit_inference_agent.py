"""
LiveKit Voice Agent — LiveKit Inference Edition
================================================
Same Airbnb agent as ``livekit_basic_agent.py``, but the STT / LLM / TTS
pipeline is provided by **LiveKit Inference** instead of direct provider
plugins.

Why this file exists for the workshop
-------------------------------------
Attendees only need a free LiveKit Cloud account (Build plan = $0, no credit
card, ~50 minutes of voice included). They do NOT need OpenAI or Deepgram
accounts. Three environment variables are enough:

    LIVEKIT_URL=wss://your-project.livekit.cloud
    LIVEKIT_API_KEY=...
    LIVEKIT_API_SECRET=...

Pedagogical points
------------------
1. The ``Assistant`` class is imported unchanged from
   ``livekit_basic_agent.py``. The agent's instructions, tools, and logic are
   **pipeline-agnostic** — only the four lines inside ``AgentSession(...)``
   change when you swap providers.

2. A post-session eval hook (``add_shutdown_callback``) runs ``JudgeGroup``
   against the conversation history whenever a session ends. Verdicts print
   to this terminal AND auto-tag the session in the LiveKit Cloud dashboard.
   This is the "live conversation → live eval" workshop centerpiece.

3. ``setup_langfuse()`` is called at the top of the entrypoint. If
   ``LANGFUSE_PUBLIC_KEY`` + ``LANGFUSE_SECRET_KEY`` are set, every
   conversation's OpenTelemetry traces also flow to LangFuse for visualization
   in their UI. Silently skipped if those env vars are missing.

Run modes
---------
    uv run python livekit_inference_agent.py console   # local mic/speakers
    uv run python livekit_inference_agent.py dev       # LiveKit Cloud, browser playground
    uv run python livekit_inference_agent.py start     # production

Environment toggles
-------------------
    RUN_LIVE_EVALS=0       # skip post-session evals (default: 1, enabled)
    LANGFUSE_PUBLIC_KEY=   # if set with LANGFUSE_SECRET_KEY, export OTel traces to LangFuse
    LANGFUSE_SECRET_KEY=
    LANGFUSE_HOST=         # optional; defaults to https://cloud.langfuse.com
"""
import base64
import os

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, RoomOutputOptions, WorkerOptions, cli, inference
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
from livekit.plugins import silero

from livekit_basic_agent import Assistant

load_dotenv(".env")
load_dotenv(".env.local")

# Toggle for the workshop demo. Set RUN_LIVE_EVALS=0 in your environment to
# skip the post-session eval (e.g. during casual iteration). Each eval costs
# roughly $0.02 in LiveKit Inference credits and adds ~13 s to session
# shutdown.
RUN_LIVE_EVALS = os.getenv("RUN_LIVE_EVALS", "1") == "1"

# Module-level holder for the LangFuse OTel TracerProvider. Stored here so the
# shutdown callback in entrypoint() can call force_flush() before the process
# exits — without an explicit flush, BatchSpanProcessor's queued spans can be
# lost when the worker shuts down faster than its 5-second flush interval.
_LANGFUSE_PROVIDER = None

# ANSI colors for the terminal output of the eval verdicts. Works in macOS
# Terminal, iTerm2, and VS Code's integrated terminal. If your terminal
# doesn't render them, set NO_COLOR=1 to disable.
if os.getenv("NO_COLOR"):
    _GREEN = _YELLOW = _RED = _CYAN = _BOLD = _RESET = ""
else:
    _GREEN = "\033[92m"
    _YELLOW = "\033[93m"
    _RED = "\033[91m"
    _CYAN = "\033[96m"
    _BOLD = "\033[1m"
    _RESET = "\033[0m"


def setup_langfuse() -> None:
    """Wire LiveKit's OpenTelemetry traces to LangFuse, if credentials are set.

    Activates only when ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY``
    are both present in the environment. Silently no-ops otherwise — the
    agent runs fine without LangFuse, this is purely additive.

    Once active, every conversation produces traces visible in BOTH:
      • LiveKit Cloud's Agent Console / Insights (built-in)
      • LangFuse's UI at the configured host (free hobby plan works)

    This pattern works with any OpenTelemetry-compatible backend — swap the
    endpoint env vars for Arize, Phoenix, Honeycomb, Datadog, etc.
    """
    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not (pk and sk):
        return  # silently disabled

    try:
        from livekit.agents.telemetry import set_tracer_provider
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        auth = base64.b64encode(f"{pk}:{sk}".encode()).decode()
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = (
            f"{host.rstrip('/')}/api/public/otel"
        )
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth}"

        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        set_tracer_provider(provider)

        # Park the provider at module scope so the shutdown callback can
        # force_flush + shutdown it cleanly. Without this, BatchSpanProcessor's
        # queued spans get dropped when the worker exits before the next flush
        # interval.
        global _LANGFUSE_PROVIDER
        _LANGFUSE_PROVIDER = provider

        print(
            f"{_GREEN}{_BOLD}✓ LangFuse OTel tracing enabled → {host}{_RESET}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 — never crash the agent on telemetry setup
        print(
            f"{_YELLOW}⚠ LangFuse setup failed (continuing without): {exc}{_RESET}",
            flush=True,
        )


async def _flush_langfuse_traces() -> None:
    """Force-flush queued OTel spans before the worker exits.

    Registered as a shutdown callback in ``entrypoint``. Without this call,
    BatchSpanProcessor's queue (defaults to ~5s flush interval) can be
    discarded mid-batch when the process tears down faster than the timer.
    """
    if _LANGFUSE_PROVIDER is None:
        return
    try:
        # force_flush is synchronous and blocking — wrap in to_thread so it
        # doesn't block the asyncio event loop during shutdown.
        import asyncio

        flushed = await asyncio.to_thread(
            _LANGFUSE_PROVIDER.force_flush, 5000
        )
        if flushed:
            print(
                f"{_GREEN}✓ LangFuse traces flushed.{_RESET}", flush=True
            )
        else:
            print(
                f"{_YELLOW}⚠ LangFuse flush timed out — some spans may be missing.{_RESET}",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001
        print(
            f"{_YELLOW}⚠ LangFuse flush failed: {exc}{_RESET}", flush=True
        )


async def _run_post_session_evals(session: AgentSession) -> None:
    """Score the conversation we just had with LiveKit's built-in judges.

    Called from the JobContext shutdown callback. Prints verdicts to this
    terminal. ``JudgeGroup`` also auto-tags the session with each verdict
    (``lk.judge.<name>:<verdict>``), which surfaces in the LiveKit Cloud
    dashboard.
    """
    message_count = sum(
        1 for item in session.history.items if item.type == "message"
    )
    if message_count < 3:
        print(
            f"\n{'─' * 60}\n"
            f" Live session eval — skipped "
            f"(only {message_count} message(s); need at least 3)\n"
            f"{'─' * 60}\n"
        )
        return

    print(
        f"\n{_BOLD}{_CYAN}{'═' * 60}\n"
        f" LIVE SESSION EVAL — running 7 judges concurrently...\n"
        f"{'═' * 60}{_RESET}"
    )

    try:
        judge_llm = inference.LLM(model="openai/gpt-4.1-mini")
        judges = JudgeGroup(
            llm=judge_llm,
            judges=[
                task_completion_judge(),
                tool_use_judge(),
                accuracy_judge(),
                relevancy_judge(),
                coherence_judge(),
                conciseness_judge(),
                safety_judge(),
            ],
        )
        evaluation = await judges.evaluate(session.history)

        verdict_color = {"pass": _GREEN, "maybe": _YELLOW, "fail": _RED}
        verdict_emoji = {"pass": "PASS", "maybe": "MAYBE", "fail": "FAIL"}

        print()
        for name, judgment in evaluation.judgments.items():
            color = verdict_color.get(judgment.verdict, "")
            label = verdict_emoji.get(judgment.verdict, judgment.verdict.upper())
            print(f"  {color}{_BOLD}[{label}]{_RESET} {color}{name}{_RESET}")
            print(f"        {judgment.reasoning}\n")

        summary_color = _GREEN if evaluation.all_passed else _YELLOW
        print(
            f"{summary_color}{_BOLD}"
            f"Summary  score={evaluation.score:.2f}  "
            f"all_passed={evaluation.all_passed}"
            f"{_RESET}"
        )
        print(f"{_BOLD}{_CYAN}{'═' * 60}{_RESET}\n")

    except Exception as exc:  # noqa: BLE001 — never crash agent shutdown on eval failure
        print(f"\n{_RED}⚠  Live eval failed: {exc}{_RESET}\n")


async def entrypoint(ctx: agents.JobContext):
    """Main entry point. The whole pipeline is the four ``inference.*`` lines below."""

    # Wire OpenTelemetry → LangFuse before the session starts so traces are
    # captured from the very first event. No-ops if LANGFUSE_* env vars are
    # not set, so this is safe to leave in for everyone.
    setup_langfuse()

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="en"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        ),
        vad=silero.VAD.load(),
    )

    # Post-session eval hook. Fires when the session shuts down — the user
    # clicks "Stop session" in Agent Console, the participant disconnects,
    # Ctrl+C is pressed in console mode, etc. Runs JudgeGroup against
    # session.history and prints verdicts to this terminal. Verdicts also
    # surface in the LiveKit Cloud session dashboard as `lk.judge.*` tags.
    if RUN_LIVE_EVALS:
        async def _on_shutdown() -> None:
            await _run_post_session_evals(session)
        ctx.add_shutdown_callback(_on_shutdown)

    # Force-flush LangFuse traces last so the eval result spans (created
    # inside _run_post_session_evals above) are included in the same batch.
    # Registered after the eval callback because shutdown callbacks run in
    # registration order.
    ctx.add_shutdown_callback(_flush_langfuse_traces)

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

    await session.generate_reply(
        instructions="Greet the user warmly and ask how you can help."
    )


if __name__ == "__main__":
    # ``agent_name`` registers this worker under a known identity so it shows
    # up in the LiveKit Cloud Agents dashboard and the Agent Console can find
    # it when you run in ``dev`` mode. Pick anything unique to your project.
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="livekit-workshop-agent",
        )
    )
