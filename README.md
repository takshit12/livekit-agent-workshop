# LiveKit Voice Agent — Workshop Starter

A small, runnable starter for building realtime voice AI agents with
[LiveKit Agents](https://docs.livekit.io/agents/). Three agents are
included so you can pick the path that matches your accounts:

| File | What it shows | Credentials you need |
|---|---|---|
| `livekit_inference_agent.py` | **Recommended.** Single STT+LLM+TTS pipeline routed through LiveKit Inference. | LiveKit Cloud only |
| `livekit_basic_agent.py` | The classic Deepgram + OpenAI pipeline, plus a `get_current_date_and_time` tool. | OpenAI + Deepgram (LiveKit creds optional) |
| `livekit_mcp_agent.py` | Same pipeline, with an example MCP server wired in for tool calling. | OpenAI + Deepgram (+ a running MCP server) |

Tests live in `tests/` and use LiveKit's built-in LLM-as-judge framework
to score behavior — useful as a reference for evals in your own agents.

---

## 1. Prerequisites

- **Python 3.10+** — check with `python --version`
- **[uv](https://docs.astral.sh/uv/)** for dependency + environment management
  ```bash
  # macOS / Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Windows (PowerShell)
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- A working microphone and speakers (for console mode)

You will also need one of:

- **A LiveKit Cloud account** — free Build plan, no credit card.
  Sign up at <https://cloud.livekit.io>. (Recommended — unlocks the
  Inference path and means you don't need separate OpenAI / Deepgram keys.)
- **OpenAI + Deepgram API keys** — if you'd rather hit those providers directly.
  - OpenAI: <https://platform.openai.com/api-keys>
  - Deepgram: <https://console.deepgram.com>

---

## 2. Install dependencies

From the project root:

```bash
uv sync
```

This creates a `.venv/` and installs everything pinned in `uv.lock`.

---

## 3. Set up environment variables

Copy the template and fill in whichever path you're taking:

```bash
cp .env.example .env
```

Open `.env` and uncomment + fill in the values for your path:

- **Path A — LiveKit Inference (recommended):** set `LIVEKIT_URL`,
  `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`. The easiest way to populate
  these is via the LiveKit CLI (see step 4) — it will write them for you.
- **Path B — Direct OpenAI + Deepgram:** uncomment `OPENAI_API_KEY` and
  `DEEPGRAM_API_KEY` and paste your keys.

You can mix paths — e.g. use LiveKit Inference for the recommended agent
and also keep OpenAI/Deepgram keys around for the basic agent.

---

## 4. (Recommended) Install the LiveKit CLI and authenticate

This step is required if you're using LiveKit Cloud (Path A) or planning
to deploy. Skip if you're going pure Path B and only running locally.

**Install:**

```bash
# macOS
brew install livekit-cli

# Linux
curl -sSL https://get.livekit.io/cli | bash

# Windows
winget install LiveKit.LiveKitCLI
```

**Authenticate (opens a browser):**

```bash
lk cloud auth
```

**Write your project credentials into `.env.local`:**

```bash
lk app env -w
```

The agents auto-load both `.env` and `.env.local`, so this is enough.

---

## 5. Download the local model files

The voice pipeline ships a small Silero VAD model and a turn detector
that need to be cached locally before first run:

```bash
uv run python livekit_inference_agent.py download-files
# (or whichever agent you plan to run)
```

You only need to do this once per machine.

---

## 6. Run an agent in console mode

Console mode talks to your local microphone and speakers — no browser,
no LiveKit room, no deployment. Best way to verify the setup works.

```bash
# Recommended — LiveKit Inference path
uv run python livekit_inference_agent.py console

# Or the classic OpenAI + Deepgram pipeline
uv run python livekit_basic_agent.py console
```

Speak into your microphone and the agent will respond out loud. Hit
`Ctrl+C` to stop.

---

## 7. (Optional) Run in a browser via LiveKit Cloud

If you want to test the agent over WebRTC the same way real users would:

```bash
# Start your agent connected to your LiveKit Cloud project
uv run python livekit_inference_agent.py dev
```

Then open the [LiveKit Agents Playground](https://agents-playground.livekit.io/),
sign in to your project, and start a session. Your local agent will pick
up the room.

---

## 8. Demo: perceived-latency improvement (A/B)

Voice agents *feel* slow when a tool call leaves the user in silence for
two or three seconds. The agents in this repo mitigate that two ways:

1. **Prompt-driven** — the model is told to emit a brief natural
   acknowledgment ("let me check that", "one sec") before slow tool calls.
   Its filler text gets TTS'd *concurrently* with tool execution, so the
   perceived gap collapses to ~zero.
2. **Programmatic backup** — `search_the_web` also fires `session.say(...)`
   with a varied hold message, as a deterministic safety net for the
   slowest path.

Both behaviors are gated on the `ACKNOWLEDGE_TOOL_CALLS` environment
variable so you can A/B them live:

```bash
# OFF — the audience hears the raw silent gap
ACKNOWLEDGE_TOOL_CALLS=0 uv run python livekit_inference_agent.py console

# ON (default) — the gap is masked by a natural-sounding acknowledgment
ACKNOWLEDGE_TOOL_CALLS=1 uv run python livekit_inference_agent.py console
```

Suggested demo script:

1. Start with `ACKNOWLEDGE_TOOL_CALLS=0`. Ask *"what's the weather in
   Tokyo today"* — point out the 2-3 second silence while DuckDuckGo runs.
2. `Ctrl+C`, relaunch with `ACKNOWLEDGE_TOOL_CALLS=1`. Ask the same
   question — the agent now acknowledges and the gap collapses.
3. Optional: keep a stopwatch visible on slides for the timing payoff.

The variable defaults to `1`, so attendees who don't touch it get the
better UX out of the box.

---

## 9. Run the tests

The test suite uses LiveKit's text-mode test harness — no audio, no
LiveKit room. Each test costs roughly a fraction of a cent in LLM usage.

```bash
uv run pytest -v
```

Useful subsets:

```bash
# Behavioral tests for the basic agent
uv run pytest -v tests/test_basic_agent.py

# Eval tests with verbose judge output
LIVEKIT_EVALS_VERBOSE=1 uv run pytest -v -s tests/test_evals.py
```

Tests are skipped automatically if you don't have either LiveKit
Inference credentials or an `OPENAI_API_KEY` set.

---

## Project layout

```
livekit-agent-workshop/
├── livekit_inference_agent.py   # Recommended starter — LiveKit Inference
├── livekit_basic_agent.py       # OpenAI + Deepgram pipeline + sample tool
├── livekit_mcp_agent.py         # Same pipeline, with MCP server tools
├── pyproject.toml               # Dependencies (managed via uv)
├── uv.lock                      # Locked dependency versions
├── .env.example                 # Template for environment variables
└── tests/
    ├── conftest.py              # Shared LLM credentials + skip logic
    ├── test_basic_agent.py      # Behavioral tests
    ├── test_evals.py            # LLM-as-judge evals
    └── test_mcp_agent.py        # MCP agent tests
```

---

## Troubleshooting

- **`Command not found: uv`** — re-open your terminal after installing, or
  add `~/.local/bin` (macOS/Linux) to your `PATH`.
- **"No LLM credentials found" when running tests** — set either
  `LIVEKIT_URL` + `LIVEKIT_API_KEY` + `LIVEKIT_API_SECRET`, **or**
  `OPENAI_API_KEY`, in `.env` or `.env.local`.
- **Agent doesn't hear you in console mode** — check OS microphone
  permissions for your terminal app. On macOS: System Settings →
  Privacy & Security → Microphone.
- **First run is slow** — model downloads happen on first invocation. Run
  the `download-files` command in step 5 to do this up front.
- **Python is too old** — `pyproject.toml` requires Python ≥ 3.10. Install
  a newer Python (`brew install python@3.12`, `pyenv install 3.12`, etc.).

---

## Resources

- LiveKit Agents docs — <https://docs.livekit.io/agents/>
- LiveKit Python SDK — <https://github.com/livekit/agents>
- Agents Playground — <https://agents-playground.livekit.io/>
- LiveKit Inference — <https://docs.livekit.io/agents/inference/>
- Model Context Protocol — <https://modelcontextprotocol.io/>
