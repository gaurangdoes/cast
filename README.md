# Dextra

A voice assistant on a Vexo wristband that knows **when you are talking to it** and **who is talking**, and schedules tasks from plain speech.

Raise the band to your mouth and it acts on what you say. Leave your arm down and it only listens. It tells the people in a conversation apart, learns their names, and can run tasks later: at a time, on a repeat, when something is said, or when you get up.

Built at VexoHacks (Y Combinator, 14 November 2026) on the [Vexo Open kit](https://github.com/VexoAI-Org/vexo-open-kit).

## What it does

| You do | Dextra does |
|---|---|
| Raise the band to your mouth and speak | Treats it as a command or a question |
| Talk with your arm down | Remembers it, with who said it. Never acts on it. |
| Double-tap the band, or close your fist | Turns the microphone off (the band's blue light goes out). Again to resume. |
| Tap the band | Acknowledges a task that just went off |

Things you can say with the band at your mouth:

- "I just parked and it ends in 2 hours, remind me 5 minutes before."
- "Every 20 minutes, tell me what I've promised people."
- "In 10 minutes, recap what we decided."
- "When Hardik mentions the deadline, tell me how long is left."
- "When I get up, remind me to take my charger."
- "What did Hardik just say?"
- "That was Hardik." (names the last person who spoke)
- "Cancel the parking one." / "Cancel all my tasks."

## How it works

```
Vexo band (Bluetooth)
 ├─ motion, 100/s (accelerometer + gyroscope) ─► wrist pose · fist detector
 ├─ taps (band firmware) ─────────────────────► pause / acknowledge
 ├─ activity (band firmware) ─────────────────► "when I get up" tasks
 └─ microphone (Opus, 16 kHz) ─► decoded once ─┬─► Deepgram: words with timestamps
                                               └─► rolling 3-minute audio buffer
                                                        │
   for each finished stretch of speech:                 ▼
     NVIDIA Nemotron 3 Diarization (local ONNX)  → where the speaker changes
     voiceprints: ECAPA + TitaNet-Large + ERes2NetV2 → which enrolled voice, or "Someone"
     loudness at the band + wrist pose + whose voice → was this said to Dextra?
                                                        │
   said to Dextra ──► Gemini 4 ──► task · question · name a speaker · cancel
   background   ──► transcript memory ──► "when someone mentions…" tasks
                                                        │
                                                        ▼
             spoken reply (Gemini speech) · React screen over a WebSocket
```

### Knowing when you are talking to it

A line counts as a command when any of these holds:

- it is loud at the band (someone is speaking right into it),
- the wrist is raised and the voice is the wearer's, even if softer,
- the wearer's recognised voice is fairly loud and the request turns out to be something Dextra can do,
- a command is already in progress (slow speech arrives in pieces).

Dextra waits for 1.8 seconds of silence before acting, so a pause mid-sentence does not cut a command in half. A command in a voice recognised as someone else's is refused.

### Knowing who is talking

Each person records about 40 seconds through the band once. After that, every stretch of speech is compared with the enrolled voices by three speaker models, and Dextra commits to a label only when one voice has a clear lead. Otherwise the line is labelled "Someone".

- While a person keeps talking, they are judged on the whole turn (up to 8 seconds), not the latest fragment.
- A voice that talks for 5 seconds and matches nobody enrolled becomes a new person automatically ("Person 3").
- Names come from conversation ("hey Hardik…", "I'm Maya") or from a command ("that was Hardik").
- Faint speech from unrecognised voices is dropped as room noise.

### Tasks

A task is a trigger plus an action.

| Trigger | Example |
|---|---|
| A time or delay | "at 4", "in 20 minutes" |
| Repeating | "every 30 minutes" |
| Something said | "when anyone mentions the deadline" |
| Movement | "when I get up" |

| Action | What happens |
|---|---|
| Say | Speaks the reminder and repeats it until you tap |
| Answer | Works out a reply from the conversation at that moment, and says it once |

Tasks are saved to `data/reminders.json` and survive a restart.

## Measured on this band

These numbers come from recordings made through the band at the event, with two real speakers. The samples are small (under 160 clips), so treat them as indicative.

**Voiceprint models**, enrolled on one recording and tested on another:

| Model | Right on 1.5 s clips | Right on 3 s clips |
|---|---|---|
| ECAPA-TDNN | 83% | 84% |
| NVIDIA TitaNet-Large | 86% | 92% |
| ERes2NetV2 | 82% | 91% |

**The three together, with the clear-lead rule:**

| Clip length | Right when it commits | How often it commits |
|---|---|---|
| 1.5 s | 96% | 74% |
| 3 s | 98% | 83% |

**Telling a new voice from a known one** needs about 5 seconds of speech: at 3 seconds the two overlap, at 5 seconds they separate.

**Loudness at the band** (dB below full scale, smaller is louder): the wearer at the mouth 12, the wearer with the arm resting 21, another person a metre away 22 to 28.

**Nemotron speaker-change detection**, on a conversation spliced from the two recordings: it separated the speakers at 68% of real changes and wrongly split 25% of single-speaker turns. Spliced audio has abrupt cuts, which is harder than real conversation.

## What does not work well

- **Short lines.** Under about 1.5 seconds there is often not enough audio to tell who spoke.
- **More than a few speakers.** Everything was measured with two enrolled voices. Accuracy will fall as the group grows.
- **Fist detection.** It works in the arm positions it was recorded in (about 7 in 10) and not in new ones. Double-tap is the dependable way to pause. Thumbs-up detection was tried and removed.
- **Wrist pose** depends on where the band sits on the wrist. If the band shifts, the poses need recording again.
- **One microphone, no noise cancellation.** Other people talking nearby are the main source of error.
- **Bluetooth in a crowded room.** Sending motion faster than 100 readings a second alongside the microphone made the band drop data.
- **Task triggers on speech and movement** were tested on their own, not yet over a long live session.

## Setup

You need Python 3.13, [uv](https://docs.astral.sh/uv/), Node with pnpm, a Vexo Open band, and keys for Deepgram and Gemini.

```bash
git clone https://github.com/gaurangdoes/cast.git && cd cast
git clone https://github.com/VexoAI-Org/vexo-open-kit.git

uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e .      # dependencies are listed in pyproject.toml

cp .env.example .env        # then fill in DEEPGRAM_API_KEY and GEMINI_API_KEY

cd web && pnpm install && pnpm build && cd ..
```

The speaker and diarization models download themselves on first run (about 600 MB) into `data/models/`.

## Calibrate (once per wearer, and again if the band moves)

```bash
.venv/bin/python -m cast.calibration <band number>
```

Open <http://localhost:8000/#calibrate>, double-tap the band so its light blinks blue, and press Connect. Then record:

1. **Voices:** 40 seconds of each person talking alone. Record the wearer two or three times with the arm in different positions.
2. **Poses** (the short steps lower on the page): Mouth, Rest Table, Side.
3. **Who is talking:** the wearer and a second person at each pose. This sets the loudness cutoffs.

Stop the server with Ctrl+C before starting Dextra. The band accepts one connection at a time.

## Run

```bash
.venv/bin/python -m cast <band number>     # then open http://localhost:8000
.venv/bin/python -m cast --sim             # pretend band with the kit's sample speech, no hardware
```

While Dextra runs, everything the band's microphone hears is sent to Deepgram. The band's light is solid blue whenever the microphone is on. Tell the people near you.

Settings, as environment variables:

| Variable | Effect |
|---|---|
| `CAST_GESTURES=0` | Turn the fist detector off |
| `CAST_FRESH=1` | Ignore voices remembered from an earlier run (only used when nobody is enrolled) |
| `CAST_MIC_GAIN=10` | Lower the microphone gain; loudness cutoffs shift to match. Untested. |
| `CAST_MODEL`, `CAST_FALLBACK_MODEL` | Gemini models for questions and for command parsing |
| `CAST_TTS=mac` | Speak with the Mac's built-in voice instead of Gemini's |
| `CAST_TTS_VOICE`, `CAST_TTS_MODEL` | Which Gemini voice and speech model to use |

To preview the screen without a band:

```bash
cd web
PORT=8799 node scripts/mock-server.mjs    # in one terminal
pnpm dev                                  # in another, then open http://localhost:5173/?ws=ws://localhost:8799
```

## Layout

```
cast/
  app.py              the main program: band events, speech pipeline, command gate, screen events
  commands.py         Gemini prompts: speech into tasks, questions answered from the transcript
  tasks.py            the task scheduler: triggers, actions, repeats, persistence
  paths.py            where data, the kit and the web build live
  hearing/
    transcribe.py     Deepgram live transcription with word timings
    diarize.py        NVIDIA Nemotron 3 Diarization through ONNX Runtime
    voices.py         enrolment, the three-model voice comparison, new-speaker discovery, learning
  motion/
    poses.py          wrist pose from the direction of gravity; loudness timeline
    gestures.py       fist detector and pose model, trained on recordings from the calibration page
  calibration/
    server.py         the calibration page's server (python -m cast.calibration)
    terminal.py       the same pose and loudness recordings from a terminal
web/                  the React screen and the calibration page (Vite + TypeScript)
data/                 recordings, models, saved tasks and voices. Never committed.
vexo-open-kit/        the band's kit, cloned here. Never committed.
```

## Built with

Vexo Open band and kit · Deepgram (speech to text) · Google Gemini (Gemini 4 for commands, questions and recaps; Gemini speech for Dextra's voice) · NVIDIA Nemotron 3 Diarization and TitaNet-Large · SpeechBrain ECAPA · ERes2NetV2 · sherpa-onnx · React and Vite.
