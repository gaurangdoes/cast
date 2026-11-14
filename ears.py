"""Deepgram live transcription with speaker labels, built on the kit's tested client.

The kit's DeepgramSpeech (vexo-open-kit/examples/python/speech.py) already handles the
connection, keep-alives and errors. We add diarize=true and read the word-level speaker
labels, so each finished piece of speech arrives as segments: who spoke, what, and when.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vexo-open-kit" / "examples" / "python"))
sys.path.insert(0, str(ROOT / "vexo-open-kit" / "python"))
import speech  # noqa: E402  (the kit's speech module)


class DiarizedDeepgram(speech.DeepgramSpeech):
    """on_segment(segment) is called for every finished stretch of speech by one speaker:
    {"speaker": int, "text": str, "start": float, "end": float}  (seconds since the stream began)
    on_partial(text) gets the live guess while someone is still talking."""

    def __init__(self, on_segment, on_partial=None, **options):
        options.setdefault("endpointing", 300)
        super().__init__(diarize="true", punctuate="true", **options)
        self.on_segment = on_segment
        self.on_partial = on_partial

    def _message(self, message):
        if message.get("type") != "Results":
            return super()._message(message)
        best = (message.get("channel", {}).get("alternatives") or [{}])[0]
        if not message.get("is_final"):
            text = str(best.get("transcript") or "").strip()
            if text and self.on_partial:
                self.on_partial(text)
            return
        words = best.get("words") or []
        for seg in group_by_speaker(words):
            self.on_segment(seg)
        # keep the kit's own bookkeeping (finals, text) working too
        super()._message(message)


def group_by_speaker(words):
    segments, cur = [], None
    for w in words:
        spk = w.get("speaker", 0)
        token = w.get("punctuated_word") or w.get("word") or ""
        if cur is None or cur["speaker"] != spk:
            if cur:
                segments.append(cur)
            cur = {"speaker": spk, "text": token, "start": float(w.get("start", 0)), "end": float(w.get("end", 0))}
        else:
            cur["text"] += " " + token
            cur["end"] = float(w.get("end", cur["end"]))
    if cur:
        segments.append(cur)
    return segments
