# Bizom Cafe Voice Feedback Agent

A LiveKit voice agent that collects feedback from Bizom employees about the cafe: food quality, menu items and suggestions. It greets the user when they join, listens, and replies briefly in a warm Indian-English tone. It stays on the topic of the cafe.

## Stack

| Stage | Provider | Model |
|-------|----------|-------|
| Speech-to-text | [Smallest AI](https://smallest.ai) | `pulse` |
| LLM | Google Gemini | `gemini-3.1-flash-lite` |
| Text-to-speech | Smallest AI | `lightning_v3.1_pro`, voice `meher` |
| Voice activity detection | Silero | tuned for a noisy cafeteria |
| Framework | [LiveKit Agents](https://docs.livekit.io/agents/) | `1.8.3` |

Turns end on voice activity (`turn_detection: "vad"`). The thresholds in `agent.py` are set so that dish clatter and background chatter don't trigger replies or interruptions.

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv venv -p 3.12
uv pip install -r requirements.txt
uv run python agent.py download-files   # fetch the Silero VAD model once
```

Create `.env` in the repo root. It's git-ignored.

```env
SMALLEST_API_KEY=...
GOOGLE_API_KEY=...        # Gemini Developer API key (not Vertex)

# Only needed to connect to a LiveKit server (dev/start modes)
LIVEKIT_URL=wss://<project>.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
```

## Run

```bash
uv run python agent.py console   # talk to the agent through your mic and speakers; no LiveKit server needed
uv run python agent.py dev       # connect to LiveKit, with hot reload
uv run python agent.py start     # production worker
```

In console mode, warnings about missing `LIVEKIT_API_KEY` are expected. Without it, the agent falls back to a local turn detector.

## Deploy

The `Dockerfile` builds a slim Python 3.13 image, pre-downloads the models and runs `agent.py start`. `livekit.toml` points at the LiveKit Cloud project and agent. Deploy with the LiveKit CLI:

```bash
lk agent deploy
```

Set `SMALLEST_API_KEY` and `GOOGLE_API_KEY` as agent secrets in LiveKit Cloud. `.env` is excluded from the image.

## Customising

All of these are in `agent.py`:

- **Menu and persona:** the `instructions` string in `VoiceAgent`. Update the daily menu there.
- **Language:** `smallestai.STT(language="en")` and `smallestai.TTS(language="en")`. Use `"hi"` for Hindi, or `"auto"` for TTS code-switching.
- **Voice:** `voice_id`. Pro voices only work with `lightning_v3.1_pro`.
- **Noise sensitivity:** `activation_threshold`, `min_speech_duration` and `min_silence_duration` in `silero.VAD.load`, plus `turn_handling` on `AgentSession`.

## Troubleshooting

**The agent joins the room but never speaks or replies.** The LLM call is almost certainly failing. Check that the Gemini model name is still served; `gemini-2.0-flash` was retired, which caused exactly this. Also check that `GOOGLE_API_KEY` is valid for the Gemini Developer API. Run `console` mode to see the error.
