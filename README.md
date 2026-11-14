# Cast

A Vexo band that reads your body language. Where your wrist is tells it what you mean:
raise it to your mouth to give a command, to your ear to ask a question, or leave it
resting and it only listens. Speaker diarization tells it who said what, and the band
works out which voice belongs to the wearer.

Built at VexoHacks (Y Combinator, 14 Nov 2026).

## Setup

    git clone <this repo> cast && cd cast
    git clone https://github.com/VexoAI-Org/vexo-open-kit.git
    uv venv --python 3.13 .venv
    uv pip install --python .venv/bin/python bleak av numpy python-dotenv websockets
    cp .env.example .env      # then fill in your keys

## Calibrate (once per wearer)

    .venv/bin/python calibrate.py <band number>     # or --sim for the pretend band

Records each wrist pose and how loud the wearer is compared with someone nearby.
Results go to `data/calibration.json`.

## Status

Work in progress during the hackathon.
