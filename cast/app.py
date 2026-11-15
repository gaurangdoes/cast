"""Dextra: the main program. One process owns the band.

    .venv/bin/python -m cast 42           # your band number
    .venv/bin/python -m cast --sim        # pretend band; plays the kit's sample speech

Then open http://localhost:8000 for the big screen.

Flow: band motion -> pose tracker; band mic -> Deepgram (with speaker labels) and a
loudness timeline. Each finished stretch of speech is tagged with its speaker, the pose
the arm was in, and a wearer score, then routed by pose:
    wrist at mouth  -> command      wrist at ear -> ask
    resting / side  -> ambient (remember only)
Double-tap = clip what was just said, credited to whoever said it.
"""
import argparse
import asyncio
import functools
import http.server
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import uuid
import time
from collections import deque
from pathlib import Path

from .paths import ROOT  # noqa: E402  (also puts the kit on the import path)

import websockets  # noqa: E402

from . import addressee  # noqa: E402
from . import commands as brain  # noqa: E402
from .hearing import transcribe as ears  # noqa: E402
from .hearing import diarize as nemotron  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from .tasks import Scheduler, keyword_hit, spoken_time  # noqa: E402
from .hearing.voices import UNKNOWN  # noqa: E402
from .motion import gestures  # noqa: E402
from .motion.poses import MODE_OF_POSE, LearnedPoses, Loudness, PoseTracker, load_calibration  # noqa: E402
from vexo_open.sync import Band  # noqa: E402

HTTP_PORT = 8000
WS_PORT = 8765
WEARER_MIN = 0.6
REFUSE_BELOW = 0.35     # only refuse when fairly sure it's someone else
COMMAND_SETTLE_FAST_S = 0.7  # ...or this long, when what was said already reads as a whole sentence
COMMAND_SETTLE_S = 1.8  # wait this long after the last raised-wrist words before acting
CLIP_SECONDS = 30
NEW_PERSON_DB = 22      # a new voice must be quieter than this at the band (the wearer resting reads 21)
COMMAND_MAX_S = 8.0     # a command stops collecting this long after its first words
FIST_SWING_DEG = 20      # a fist counts only if the arm swung less than this just before it
FAINT_DB = 24           # quieter than this and not a known voice: background chatter, dropped
SOFT_DB = 18            # the wearer's recognised voice at this level: a command if it is actionable
NEAR_DB = 21            # wrist raised and the wearer's voice: softer speech still counts
RAISED_DB = 14          # at or below this loudness someone is talking into the band
NAME_HINT = re.compile(r"\b(hey|hi|hello|yo|thanks|thank you|i'm|i am|this is|my name)\b", re.I)
CONTEXT_S = 6.0          # seconds of earlier audio given to Nemotron with each sentence


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip())


class Screen:
    """Broadcasts JSON events to every open big-screen page."""

    def __init__(self):
        self.loop = None
        self.clients = set()
        self.backlog = deque(maxlen=200)
        self.snapshot = lambda: []      # extra messages for a newly opened page
        self.on_message = None          # called with each JSON message a page sends

    def send(self, kind, **data):
        msg = json.dumps({"kind": kind, "t": time.time(), **data})
        if kind in ("segment", "clip", "action"):
            self.backlog.append(msg)
        if self.loop and not self.loop.is_closed():
            try:
                self.loop.call_soon_threadsafe(self._broadcast, msg)
            except RuntimeError:
                pass

    def _broadcast(self, msg):
        for ws in list(self.clients):
            asyncio.ensure_future(self._safe_send(ws, msg))

    async def _safe_send(self, ws, msg):
        try:
            await ws.send(msg)
        except Exception:
            self.clients.discard(ws)

    async def handler(self, ws):
        self.clients.add(ws)
        for msg in list(self.backlog) + list(self.snapshot()):
            await ws.send(msg)
        try:
            async for raw in ws:                    # the screen can send a few requests back
                try:
                    msg = json.loads(raw)
                    if isinstance(msg, dict) and self.on_message:
                        self.on_message(msg)
                except Exception as e:  # noqa: BLE001
                    print("[screen] bad message:", e)
        finally:
            self.clients.discard(ws)


class Voice:
    """Spoken replies in a Gemini voice (gemini-3.8-flash-lite-tts, about 1.6 s to generate),
    falling back to the Mac's built-in voice if that fails or is slow. While Dextra talks, speech
    the band hears is ignored so it does not transcribe itself. Replies play one at a time."""
    MODEL = os.environ.get("CAST_TTS_MODEL", "gemini-3.8-flash-lite-tts")
    NAME = os.environ.get("CAST_TTS_VOICE", "Kore")

    def __init__(self):
        self.busy_until = 0.0
        self.spoke_from = float("inf")       # when the reply now (or last) playing started
        self.q = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, text):
        self.q.put(text)

    def chime(self):
        """A short sound the moment a command is accepted, so the wait for the reply is not silence."""
        subprocess.Popen(["afplay", "-v", "0.6", "/System/Library/Sounds/Tink.aiff"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def prewarm(self, phrases):
        """Generate the fixed replies ahead of time; they then play with no delay."""
        def run():
            for text in phrases:
                try:
                    self._gemini(text)
                except Exception:  # noqa: BLE001
                    return
        if os.environ.get("CAST_TTS", "gemini") == "gemini":
            threading.Thread(target=run, daemon=True).start()

    def _run(self):
        while True:
            text = self.q.get()
            try:
                if os.environ.get("CAST_TTS", "gemini") == "gemini":
                    self._speak_gemini(text)
                else:
                    self._speak_mac(text)
            except Exception as e:  # noqa: BLE001
                print("[voice] Gemini speech failed, using the Mac voice:", str(e)[:100])
                try:
                    self._speak_mac(text)
                except Exception as e2:  # noqa: BLE001
                    print("[voice] playback failed:", e2)
            self.busy_until = max(self.busy_until, time.time() + 0.4)

    def _speak_mac(self, text):
        self.spoke_from = time.time() - 0.2
        self.busy_until = time.time() + 0.5 + len(text) * 0.07
        subprocess.run(["say", "-r", "205", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _cache_path(self, text):
        import hashlib
        cache = ROOT / "data" / "tts"
        cache.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(("%s|%s|%s" % (self.MODEL, self.NAME, text)).encode()).hexdigest()[:16]
        return cache / (key + ".pcm")                    # 16-bit mono at 24 kHz

    def _stream(self, text):
        """Gemini speech as it is generated: the first audio arrives after about half a second
        (measured), the rest within about a second. Yields raw 16-bit 24 kHz chunks."""
        import base64
        import urllib.request
        body = {"contents": [{"parts": [{"text": text}]}],
                "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self.NAME}}}}}
        req = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/models/%s:streamGenerateContent?alt=sse" % self.MODEL,
            data=json.dumps(body).encode(),
            headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=6) as r:
            for line in r:
                if not line.startswith(b"data:"):
                    continue
                for part in json.loads(line[5:]).get("candidates", [{}])[0].get("content", {}).get("parts", []):
                    data = part.get("inlineData", {}).get("data")
                    if data:
                        yield base64.b64decode(data)

    def _gemini(self, text):
        """Generate and cache a reply without playing it (used to prepare the fixed replies)."""
        path = self._cache_path(text)
        if not path.exists():
            pcm = b"".join(self._stream(text))
            if pcm:
                path.write_bytes(pcm)
        return str(path), path.stat().st_size / 48000.0

    def _speak_gemini(self, text):
        import sounddevice as sd
        path = self._cache_path(text)
        chunks = [path.read_bytes()] if path.exists() else self._stream(text)   # said before: no wait at all
        got = bytearray()
        out = None
        try:
            for chunk in chunks:
                if out is None:                          # open the speaker when the first audio arrives
                    out = sd.RawOutputStream(samplerate=24000, channels=1, dtype="int16")
                    out.start()
                    self.spoke_from = time.time() - 0.2
                got += chunk
                self.busy_until = self.spoke_from + len(got) / 48000.0 + 0.8
                out.write(bytes(chunk))
            if out is None:
                raise RuntimeError("no audio came back")
            left = self.spoke_from + 0.2 + len(got) / 48000.0 - time.time()
            if left > 0:
                time.sleep(left + 0.15)                  # let the last of it play out
        finally:
            if out is not None:
                out.stop()
                out.close()
        if not path.exists() and got:
            path.write_bytes(bytes(got))

    def speaking(self, t_start, t_end):
        """Did this stretch of heard speech overlap Dextra's own voice?"""
        return t_end >= self.spoke_from and t_start <= self.busy_until


class Dextra:
    def __init__(self, band, screen, voice):
        self.band, self.screen, self.voice = band, screen, voice
        cal = load_calibration()
        self.poses = LearnedPoses() if gestures.POSE_MODEL.exists() else PoseTracker(cal)
        # Fist / thumbs up work in the arm positions they were recorded in (about 7 in 10 unseen
        # reps), not in new ones. A fist only counts when the arm was still just before it.
        use = os.environ.get("CAST_GESTURES", "1") == "1" and gestures.MODEL.exists()
        self.gest = gestures.LiveDetector(min_prob=0.65, refractory_s=2.5) if use else None
        self.fast = self.gest is not None or isinstance(self.poses, LearnedPoses)
        self.paused, self.pause_started = False, 0.0
        self.unacked = []                     # (reminder, threading.Event) pairs waiting for a tap
        self.loud = Loudness(cal)
        self.segments = deque(maxlen=500)
        self.audio_t0 = None
        self.last_pose_sent = None
        self.last_motion_push = 0.0
        self.pool = ThreadPoolExecutor(max_workers=6)
        self.scheduler = Scheduler(self.on_reminder_due)
        self.last_glance_status = 0.0
        self.activity_ok = False              # set once the band's activity stream is on
        self.moving, self.move_last, self.move_run = False, False, 0
        print("Loading the voice model...")
        from vexo_open.audio import PcmDecoder
        from .hearing.voices import AudioBuffer, Voices
        self.voices = Voices()               # SpeechBrain ECAPA voiceprints (CPU)
        self.voices_path = ROOT / "data" / "voices.npz"
        self.lab_dir = ROOT / "data" / "voices_lab"
        if self.voices.enroll_folder(self.lab_dir):
            self.learned_path = ROOT / "data" / "voices_learned.pkl"
            n = 0 if os.environ.get("CAST_FRESH") == "1" else self.voices.load_learned(self.learned_path)
            print("Enrolled voices: you + %s%s" % (", ".join(self.voices.names.values()),
                                                  " (+%d examples learned earlier)" % n if n else ""))
        elif os.environ.get("CAST_FRESH") != "1" and self.voices.load(self.voices_path):
            print("Remembered %d voices from last time%s." % (
                len(self.voices.speakers), ", including the wearer" if self.voices.wearer_id is not None else ""))
        self.pending_name = None              # (name, time): the wearer just addressed someone by name
        threading.Thread(target=self._save_loop, daemon=True).start()
        self.audio = AudioBuffer()
        self.decoder = PcmDecoder()
        # CAST_DIARIZE=stream: Nemotron runs continuously and keeps one id per person for the
        # whole session, instead of looking at each sentence's window on its own
        self.stream, self.stream_votes = None, {}
        if os.environ.get("CAST_DIARIZE") == "stream":
            self.stream = nemotron.StreamingDiarizer()
        print("Speaker changes: %s" % ("streaming Nemotron with a session speaker cache" if self.stream
                                       else "Nemotron per sentence (CAST_DIARIZE=stream for streaming)"))
        self.last_voice = None
        self.last_run_end = None
        self.arm = deque(maxlen=200)                 # (time, ax, ay, az) about 20 a second: was the arm steady?
        # "was this said to Dextra?": the trained model if there is one (python -m cast.addressee), else the rules
        self.model_gate = os.environ.get("CAST_ADDRESSEE") != "rules" and addressee.available()
        print("Command gate: %s" % ("trained model (data/addressee_model.pkl)" if self.model_gate
                                    else "hand-written rules (no trained model yet)"
                                    if os.environ.get("CAST_ADDRESSEE") != "rules" else "hand-written rules (CAST_ADDRESSEE=rules)"))
        self.label_mode, self.label_until, self.label_session = None, 0.0, None   # a labelling session, if one is running
        self.turn = None                      # the stretch of continuous speech being labelled
        self.last_seq = None
        self.pending_cmd = None               # raised-wrist speech collected until the speaker pauses
        self.last_partial = 0.0
        self.cmd_lock = threading.Lock()
        threading.Thread(target=self._command_loop, daemon=True).start()
        self.seg_q = queue.Queue()
        threading.Thread(target=self._segment_loop, daemon=True).start()
        screen.on_message = self.on_screen_message
        screen.snapshot = lambda: [self.reminders_msg(), self.names_msg(), self.labelling_msg(), json.dumps({"kind": "mic", "t": time.time(), "on": not self.paused})]

    # ---- band events (run on the band's handler thread; keep short)

    def on_motion(self, m):
        now = time.time()
        if not self.arm or now - self.arm[-1][0] >= 0.05:
            self.arm.append((now, m.ax, m.ay, m.az))
        if isinstance(self.poses, LearnedPoses):
            pose = self.poses.feed(m.ax, m.ay, m.az, m.t, now)
        else:
            pose = self.poses.feed(m.ax, m.ay, m.az)
        if self.gest is not None:
            hit = self.gest.feed(m.t, m.ax, m.ay, m.az, m.gx, m.gy, m.gz) if m.gx is not None else None
            if hit:
                self.pool.submit(self.on_gesture, *hit)
        if pose != self.last_pose_sent or now - self.last_motion_push > 0.5:
            self.last_pose_sent, self.last_motion_push = pose, now
            self.screen.send("pose", pose=pose, mode=MODE_OF_POSE.get(pose, "ambient"),
                             angle=round(self.poses.last_angle, 1) if self.poses.last_angle is not None else None)

    def on_mic(self, e):
        rms = getattr(e, "rms_db", None)
        if rms is not None:
            self.loud.feed(rms)
            self.screen.send("level", rms_db=rms)

    def on_audio(self, a):
        if self.audio_t0 is None:
            self.audio_t0 = time.time()
        seq = getattr(a, "seq", None)
        if seq is not None and self.last_seq is not None and seq < self.last_seq - 1000:
            self.new_decoder()                       # the band restarted its audio stream
        self.last_seq = seq
        try:
            pcm = self.decoder.feed(a)               # decode once: lost packets are filled in here,
            if pcm:                                  # so Deepgram's clock and our buffer stay in step
                self.ears.feed_pcm(pcm)
                self.audio.add(pcm)
                if self.stream is not None:
                    self.stream.feed(pcm)            # same audio, so the same clock as the buffer
        except Exception as e:  # noqa: BLE001 - a bad packet must not stop transcription
            self.audio_errors = getattr(self, "audio_errors", 0) + 1
            if self.audio_errors in (1, 50):
                print("[audio] feed failed:", e)

    def new_decoder(self):
        from vexo_open.audio import PcmDecoder
        self.decoder = PcmDecoder()
        self.last_seq = None

    def on_notice(self, n):
        kind = n.get("kind") if isinstance(n, dict) else getattr(n, "kind", n)
        self.notices = getattr(self, "notices", 0) + 1
        if self.notices in (1, 10, 100) or self.notices % 500 == 0:
            print("[band notice] %s (x%d)" % (kind, self.notices))

    def on_double_tap(self, e):
        self.pool.submit(self.toggle_pause)       # firmware tap detection: the reliable way to pause

    def on_tap(self, e):
        self.pool.submit(self.acknowledge, "Tap")

    # ---- speech

    def on_partial(self, text):
        self.last_partial = time.time()
        self.screen.send("partial", text=text)

    def on_segment(self, words):
        """A finished stretch of speech from Deepgram. This runs on Deepgram's reader thread, so
        it only hands the words over: working out who spoke takes tens to hundreds of
        milliseconds per line, and doing it here would make the transcript fall behind."""
        self.seg_q.put(words)
        waiting = self.seg_q.qsize()
        if waiting >= 8 and waiting % 8 == 0:
            print("[segment] %d lines waiting to be labelled" % waiting)

    def _segment_loop(self):
        """One worker labels lines in the order they were said (the turn logic depends on order)."""
        while True:
            self.process_segment(self.seg_q.get())

    def process_segment(self, words):
        """Nemotron splits the words into turns (who spoke when, even mid-sentence); each turn's
        voiceprint gives it a stable speaker id."""
        try:
            t0 = self.audio_t0 or time.time()
            # Dextra's own spoken reply reaches the microphone too. Drop only the words that fall
            # inside the time it was speaking: dropping the whole line lost everything a person
            # said around a reply (lines are longer now that Deepgram waits 500 ms to end one).
            kept = [w for w in words if not self.voice.speaking(t0 + (w["start"] + w["end"]) / 2,
                                                                t0 + (w["start"] + w["end"]) / 2)]
            if len(kept) < len(words):
                print("[own voice] left out %d of %d words heard while Dextra was speaking: %s" % (
                    len(words) - len(kept), len(words), " ".join(w["text"] for w in words if w not in kept)[:80]))
            words = kept
            if not words:
                return
            for run in self.split_by_speaker(words):
                self.handle_run(run)
        except Exception:  # noqa: BLE001 - an error here must be seen, and must not stop the worker
            import traceback
            print("[segment] failed:")
            traceback.print_exc()

    def split_by_speaker(self, words):
        w0, w1 = words[0]["start"], words[-1]["end"]
        win_start = max(0.0, w0 - CONTEXT_S)                 # extra audio before helps Nemotron
        pcm = self.audio.slice(win_start, w1 + 0.15)
        turns = []
        if self.stream is not None:
            try:
                # the newest second has no look-ahead yet: take the model's first estimate of it
                turns = self.stream.segments(max(0.0, w0 - 1.0), w1 + 0.15, provisional=True)
            except Exception as e:  # noqa: BLE001
                print("[nemotron stream] failed:", e)
        elif len(pcm) > 16000:                                 # at least 0.5 s
            try:
                turns = [(spk, win_start + a, win_start + b) for spk, a, b in nemotron.diarize(pcm)]
            except Exception as e:  # noqa: BLE001
                print("[nemotron] failed:", e)
        # local Nemotron speaker -> global voice id, from all of that speaker's audio in the window
        def speaker_of(w):
            mid = (w["start"] + w["end"]) / 2
            best, gap = None, 1e9
            for spk, a, b in turns:
                d = 0 if a <= mid <= b else min(abs(mid - a), abs(mid - b))
                if d < gap:
                    best, gap = spk, d
            return best if gap < 1.0 else None

        runs = []
        for w in words:
            spk = speaker_of(w)
            if runs and runs[-1]["local"] == spk:
                runs[-1]["words"].append(w)
            else:
                runs.append({"local": spk, "words": [w]})
        # A piece under a second cannot be told apart by voice, and the speaker tracker sometimes
        # flips for a word or two mid-sentence ("Can you / actually, / turn the timer on"). Fold
        # such pieces into the neighbour they sit closest to, so the sentence is judged whole.
        def length(r):
            return r["words"][-1]["end"] - r["words"][0]["start"]
        while len(runs) > 1:
            i = min(range(len(runs)), key=lambda k: length(runs[k]))
            if length(runs[i]) >= 1.0:
                break
            before = runs[i]["words"][0]["start"] - runs[i - 1]["words"][-1]["end"] if i > 0 else 1e9
            after = runs[i + 1]["words"][0]["start"] - runs[i]["words"][-1]["end"] if i + 1 < len(runs) else 1e9
            if before <= after:
                runs[i - 1]["words"] += runs[i]["words"]
            else:
                runs[i + 1]["words"] = runs[i]["words"] + runs[i + 1]["words"]
            del runs[i]
        out = []
        for r in runs:
            start, end = r["words"][0]["start"], r["words"][-1]["end"]
            # Nemotron only decides where the speaker changes; who it is comes from this
            # stretch's own audio, never from the context before it
            vid, sim = self.voices.identify(self.audio.slice(start - 0.05, end + 0.05))
            stranger = vid is not None and bool(getattr(self.voices, "last_stranger", False))
            judged = vid is not None                 # a second or more of audio: the voice models had a say
            if self.stream is not None and r["local"] is not None:
                # streaming ids mean the same person all session: remember which voice each one
                # turned out to be, and use that when this stretch is too short to tell
                votes = self.stream_votes.setdefault(r["local"], {})
                if vid is not None and vid >= 0:
                    votes[vid] = votes.get(vid, 0.0) + (end - start)
                elif votes:
                    top, total = max(votes, key=votes.get), sum(votes.values())
                    if votes[top] >= 4.0 and votes[top] >= 0.8 * total:
                        vid = top
            # Someone still talking (under 0.7 s since their last words, and Nemotron saw no
            # change of speaker) is judged on everything they have said in this turn: longer
            # audio is what makes a voiceprint reliable (81% right at 1 s, 95% at 3 s).
            t = self.turn
            gap = start - t["end"] if t is not None else 9.0
            changed = vid is not None and vid >= 0 and t is not None and vid != t["vid"]
            if vid is None:
                # under a second: only a mid-sentence fragment (no real gap) is the same person
                continuing = t is not None and r is runs[0] and gap < 0.3
            else:
                continuing = t is not None and r is runs[0] and gap < 0.7
            if stranger:
                continuing = False                   # a voice Dextra has not met never inherits the last speaker's name
            if continuing and not changed:
                jv, js = self.voices.identify(self.audio.slice(max(t["start"], end - 8.0), end + 0.05))
                if jv is not None and jv >= 0:
                    vid, sim = jv, js
                elif t["vid"] >= 0:
                    vid = t["vid"]                           # still undecided: keep who it was
                t["end"] = end
                if vid is not None and vid >= 0:
                    t["vid"] = vid
            else:
                self.turn = {"start": start, "end": end, "vid": vid if vid is not None else UNKNOWN}
            if vid is None:
                vid = UNKNOWN
            if vid == UNKNOWN and end - self.turn["start"] >= self.turn.get("try_at", 3.0):
                self.turn["try_at"] = 5.0 if self.turn.get("try_at", 3.0) < 5.0 else 1e9
                # three seconds of a clear stranger, or five of someone nobody recognises: a new person
                t0 = self.audio_t0 or time.time()
                level = self.loud.level_during(t0 + self.turn["start"], t0 + end)
                if level is not None and level - getattr(self, "db_shift", 0) > NEW_PERSON_DB:   # quieter than the wearer ever is
                    nid = self.voices.discover(self.audio.slice(max(self.turn["start"], end - 8.0), end + 0.05))
                    if nid is not None:
                        vid = self.turn["vid"] = nid
                        print("[voices] new person: %s" % self.voices.names[nid])
                        self.screen.send("names", names={str(k): v for k, v in self.voices.names.items()})
                        self.card(uuid.uuid4().hex[:8], "done", "command", "New voice",
                                  result="%s joined the conversation" % self.voices.names[nid])
            self.last_run_end = end
            vid = self.voices.resolve(vid)
            self.last_voice = vid
            guess = None                             # not sure enough to commit: still say who it sounds most like
            # Never guess the wearer: only a committed match to the wearer's voice counts as them,
            # or a look-alike voice could borrow their name. The guess is for display only.
            if (vid == UNKNOWN and judged and not stranger and getattr(self.voices, "last_lead", 0) >= 0.04
                    and self.voices.last_top != self.voices.wearer_id):
                guess = self.voices.last_top
            out.append({"voice": vid, "sim": sim, "start": start, "end": end, "stranger": stranger and vid == UNKNOWN,
                        "guess": guess,
                        "text": " ".join(w["text"] for w in r["words"])})
        return out

    def handle_run(self, run):
        t0 = self.audio_t0 or time.time()
        start, end = t0 + run["start"], t0 + run["end"]
        pose = self.poses.pose_during(start, end)
        measured_pose, raised_angle = pose, self.poses.angle_to("mouth", start, end)
        rms = self.loud.level_during(start, end)
        if rms is not None:
            rms -= getattr(self, "db_shift", 0)      # as if the mic were at the gain we calibrated with
        vid, sim = run["voice"], run["sim"]
        # Who is being talked to? Three signals, measured on this band: loudness at the mic
        # (12 dB at the mouth, 21 resting, 27 someone else), the wrist pose, and whose voice it is.
        loud = rms is not None and rms <= RAISED_DB                 # anyone speaking right into the band
        stranger = bool(run.get("stranger"))                        # sounds like nobody Dextra knows
        if vid < 0 and loud and self.voices.wearer_id is not None and not stranger:
            vid = self.voices.wearer_id       # undecided voice, but this loud at the band is the wearer
        if stranger and loud:
            print("[command] refused, not a voice Dextra knows: %s" % run["text"][:70])
            self.card(uuid.uuid4().hex[:8], "refused", "command", run["text"][:80],
                      reason="That was not your voice. Dextra only takes instructions from its wearer.")
            loud = False
        mine = vid == self.voices.wearer_id                         # recognised as the wearer
        near = rms is not None and rms <= NEAR_DB and pose == "mouth" and mine   # softer, wrist raised
        with self.cmd_lock:
            # still finishing a slow command: the wearer, or an undecided voice that is close to the band
            pending_now = self.pending_cmd is not None
            ongoing = pending_now and not stranger and (
                mine or (vid < 0 and rms is not None and rms <= NEAR_DB))
        # the wearer's own voice, fairly loud, but the pose is not clearly raised: treat it as a
        # command only if it turns out to be something Dextra can do (decided after Gemini reads it)
        soft = (not loud and not near and not ongoing and vid == self.voices.wearer_id
                and rms is not None and rms <= SOFT_DB)
        # What the gate knew when it decided, kept (without audio) so a model can be trained on it.
        voice_kind = "wearer" if mine else ("other" if vid >= 0 and self.voices.wearer_id is not None else "unknown")
        row = {"t": round(start, 2), "text": run["text"], "duration": round(end - start, 2),
               "words": len(run["text"].split()), "rms_db": rms, "pose": measured_pose, "pose_angle": raised_angle,
               "voice": voice_kind, "margin": round(float(sim or 0), 3), "pending": pending_now,
               "since_cast": round(max(0.0, start - self.voice.busy_until), 1) if self.voice.busy_until else None,
               "rule": {"loud": bool(loud), "near": bool(near), "soft": bool(soft), "ongoing": bool(ongoing),
                        "to_cast": bool(loud or near or ongoing or soft)}}
        prob = None
        if self.model_gate and not loud:             # speech right at the band is always to Dextra
            try:
                prob = addressee.predict(row)
            except Exception as e:  # noqa: BLE001
                print("[gate] model failed, using the rules:", e)
            if prob is not None:
                if voice_kind == "other":            # a recognised other person never commands Dextra
                    near = ongoing = soft = False
                else:
                    near, soft, ongoing = prob >= 0.75, 0.5 <= prob < 0.75, False
        labelling = self.labelling()
        row.update(prob=round(prob, 3) if prob is not None else None, session=self.label_session if labelling else None,
                   label=(None if not labelling else 0 if labelling == "chat" else (1 if voice_kind != "other" else None)))
        try:
            addressee.log(row)
        except Exception as e:  # noqa: BLE001
            print("[gate] could not log:", e)
        if (rms is not None and rms <= RAISED_DB - 2 and vid == self.voices.wearer_id and end - start >= 1.0
                and not isinstance(self.poses, LearnedPoses)):
            # the wearer, this loud: the band is at the mouth for certain. Let the "raised" pose
            # follow where the wrist really was, so a shifted band corrects itself.
            moved = self.poses.adapt("mouth", start, end)
            if moved is not None and moved > 8:
                print("[pose] raised-wrist reference moved %.0f degrees toward how you hold it now" % moved)
        to_cast = loud or near or ongoing or soft
        if loud or near or ongoing:
            pose = "mouth"                           # shown as "to Dextra"; a soft one stays plain until it proves actionable
        elif pose == "mouth":
            pose = "rest_table"                      # wrist up, but not talking to the band
        score = self.wearer_score(vid, rms, pose)
        mode = MODE_OF_POSE.get(pose, "ambient")
        if vid < 0 and mode == "ambient" and rms is not None and rms > FAINT_DB:
            self.ignored = getattr(self, "ignored", 0) + 1      # faint and nobody we know: room noise
            self.screen.send("ignored", count=self.ignored)
            print("[noise] ignored (%s dB): %s" % (rms, run["text"][:60]))
            return
        is_wearer = vid == self.voices.wearer_id
        # "hey Hardik, ..." a moment ago + a new, unnamed voice answering = that voice is Hardik
        if (self.pending_name and not is_wearer and vid >= 0 and self.voices.wearer_id is not None
                and vid not in self.voices.names and time.time() - self.pending_name[1] < 25
                and len(run["text"].split()) >= 2):
            self.set_name(vid, self.pending_name[0], "answered when you said their name")
            self.pending_name = None
        entry = {"speaker": vid, "text": run["text"], "start": start, "end": end,
                 "pose": pose, "rms_db": rms, "wearer": score, "mode": mode,
                 "is_wearer": is_wearer, "voice_sim": round(sim, 2), "name": self.voices.names.get(vid),
                 "guess": (None if vid >= 0 or run.get("guess") is None or run["guess"] == self.voices.wearer_id
                           else self.voices.names.get(run["guess"])),
                 "id": uuid.uuid4().hex[:8], "a0": run["start"], "a1": run["end"]}
        if mode == "ambient" and NAME_HINT.search(run["text"]):
            self.pool.submit(self.learn_names, entry)
        self.segments.append(entry)
        self.screen.send("segment", **entry)
        if mode == "ambient" and self.scheduler.waiting("heard"):
            self.pool.submit(self.check_heard, entry)
        print("[voice %s%s sim %.2f | %-10s | wearer %.2f | %s] %s   {%s}" % (
            vid, "*" if entry["is_wearer"] else "", sim, pose, score, mode, run["text"],
            getattr(self.voices, "last_scores", "")))
        if os.environ.get("CAST_DEBUG_AUDIO") == "1":       # keep each line's audio to study labelling misses
            d = ROOT / "data" / "debug_audio"
            d.mkdir(exist_ok=True)
            (d / ("%07.1f_v%s.pcm" % (run["start"], vid))).write_bytes(self.audio.slice(run["start"] - 0.05, run["end"] + 0.05))
        if to_cast and not labelling:                # while examples are being collected, Dextra does not act
            with self.cmd_lock:                      # collect until the speaker pauses
                if self.pending_cmd is None:
                    self.pending_cmd = {"texts": [], "scores": [], "entry": entry, "other": [], "soft": True,
                                        "first": time.time()}
                if loud or near:
                    self.pending_cmd["soft"] = False  # said straight to the band: Dextra may answer aloud
                self.pending_cmd["texts"].append(entry["text"])
                self.pending_cmd["scores"].append(score)
                # a recognised voice that is not the wearer's
                self.pending_cmd["other"].append(
                    self.voices.wearer_id is not None and vid >= 0 and vid != self.voices.wearer_id)
                self.pending_cmd["last"] = time.time()
                said = " ".join(self.pending_cmd["texts"]).strip()
                if len(said.split()) >= 3:           # read it now; if more words come, this guess is dropped
                    tasks_now = [r["task"] for r in sorted(self.scheduler.pending(), key=lambda r: r.get("created", 0))]
                    self.pending_cmd["guess"] = (said, self.pool.submit(brain.interpret_command, said, tasks_now))

    def _command_loop(self):
        while True:
            time.sleep(0.2)
            with self.cmd_lock:
                p = self.pending_cmd
                if p is None:
                    continue
                now = time.time()
                said = " ".join(p["texts"]).strip()
                finished = said[-1:] in ".?!" and len(said.split()) >= 4      # a whole sentence
                settle = COMMAND_SETTLE_FAST_S if finished else COMMAND_SETTLE_S
                if (now - p["first"] < COMMAND_MAX_S
                        and (now - p["last"] < settle or now - self.last_partial < min(settle, 0.8))):
                    continue                         # still talking
                self.pending_cmd = None
            text = " ".join(p["texts"]).strip()
            score = max(p["scores"])
            if len(text.split()) < 3:
                print("[command] ignored fragment:", text)
                continue
            entry = dict(p["entry"], text=text, wearer=score, soft=p.get("soft", False))
            guess = p.get("guess")
            entry["parsed"] = guess[1] if guess and guess[0] == text else None
            if p["other"] and all(p["other"]):       # every part was someone else's voice
                self.card(uuid.uuid4().hex[:8], "refused", "command", text,
                          reason="Not the wearer's voice (score %.2f)" % score)
                self.voice.say("Only the wearer can do that.")
            else:
                self.act("command", entry)

    def card(self, aid, status, mode, text, result=None, reason=None):
        self.screen.send("action", id=aid, status=status, mode=mode, text=text, result=result, reason=reason)

    # ---- names

    def set_name(self, vid, name, why):
        if not name or self.voices.names.get(vid) == name:
            return
        for other, n in list(self.voices.names.items()):     # one voice per name
            if n == name and other != vid:
                del self.voices.names[other]
        self.voices.names[vid] = name
        if self.voices.enrolled:                     # remember the name with the enrolment
            try:
                given = {self.voices.labels[i]: n for i, n in self.voices.names.items() if i < len(self.voices.labels)}
                (self.lab_dir / "names.json").write_text(json.dumps(given))
            except Exception as e:  # noqa: BLE001
                print("[names] could not save:", e)
        print("[names] voice %s is %s (%s)" % (vid, name, why))
        self.screen.send("names", names={str(k): v for k, v in self.voices.names.items()})
        self.card(uuid.uuid4().hex[:8], "done", "command", "Learned a voice", result="Speaker %s is %s (%s)" % (vid, name, why))

    # ---- labelling sessions: examples for the trained command gate

    LABEL_SECONDS = {"cast": 90, "chat": 120}

    def labelling(self):
        """The running session's kind ("cast" or "chat"), or None."""
        if self.label_mode and time.time() > self.label_until:
            self.set_labelling("off")
        return self.label_mode

    def set_labelling(self, mode):
        if mode in self.LABEL_SECONDS:
            self.label_mode = mode
            self.label_until = time.time() + self.LABEL_SECONDS[mode]
            self.label_session = "%s-%d" % (mode, int(time.time()))
            session = self.label_session
            with self.cmd_lock:
                self.pending_cmd = None              # nothing half-said carries into the session

            def expire():
                time.sleep(self.LABEL_SECONDS[mode] + 0.5)
                if self.label_session == session and self.label_mode:
                    self.set_labelling("off")
            threading.Thread(target=expire, daemon=True).start()
            print("[gate] labelling session %s for %d s" % (session, self.LABEL_SECONDS[mode]))
        else:
            if self.label_mode:
                print("[gate] labelling session %s ended" % self.label_session)
            self.label_mode, self.label_until = None, 0.0
        self.screen.send("labelling", mode=self.label_mode, until=self.label_until if self.label_mode else None)

    def labelling_msg(self):
        on = self.label_mode and time.time() <= self.label_until
        return json.dumps({"kind": "labelling", "t": time.time(), "mode": self.label_mode if on else None,
                           "until": self.label_until if on else None})

    def on_screen_message(self, msg):
        """Typed on the screen: click a speaker's name and rename them."""
        if msg.get("cmd") == "label":
            self.set_labelling(str(msg.get("mode")))
        elif msg.get("cmd") == "assign":
            self.pool.submit(self.assign, str(msg.get("seg")), msg.get("name"))
        elif msg.get("cmd") == "rename":
            try:
                vid = int(msg.get("speaker"))
            except (TypeError, ValueError):
                return
            name = " ".join(str(msg.get("name") or "").split())[:30]
            if vid >= 0 and vid != self.voices.wearer_id and name:
                if self.means_wearer(name):          # a second person named "me" would compete with You
                    self.card(uuid.uuid4().hex[:8], "refused", "command", "Name a voice",
                              reason="To mark a line as yours, click that line and choose You.")
                    return
                self.pool.submit(self.set_name, vid, name, "you typed it")

    @staticmethod
    def means_wearer(name):
        """"me", "Me'", "that's me", "myself": the wearer, never a new person called "me"."""
        words = re.sub(r"[^a-z ]", "", str(name).lower().replace("'s", " ").replace("’s", " ")).split()
        return bool(words) and set(words) <= {"me", "you", "i", "myself", "mine", "my", "voice", "that", "thats",
                                             "this", "was", "is", "its", "it", "wearer"} \
            and bool(set(words) & {"me", "you", "i", "myself", "mine", "my", "wearer"})

    def assign(self, seg_id, who):
        """Typed on the screen for a line Dextra could not place: "that was me" / "Hardik" / a new
        name. The line is relabelled and its audio becomes an example of that voice, so Dextra
        recognises them next time."""
        seg = next((s for s in list(self.segments) if s.get("id") == seg_id), None)
        who = " ".join(str(who or "").split())[:30]
        if seg is None or not who:
            return
        pcm = self.audio.slice(seg["a0"] - 0.05, seg["a1"] + 0.05)      # empty if older than 3 minutes
        if self.means_wearer(who):
            vid = self.voices.wearer_id
            if vid is not None:
                self.voices.teach(pcm, vid=vid)
        else:
            vid = self.find_voice(who)
            if vid is not None:
                self.voices.teach(pcm, vid=vid)
            else:
                vid = self.voices.teach(pcm, vid=None, name=who)          # someone new
                if vid is None:
                    self.card(uuid.uuid4().hex[:8], "refused", "command", "Name a voice",
                              reason="That line is too short to learn a new voice from. Try a longer one.")
                    return
                self.screen.send("names", names={str(k): v for k, v in self.voices.names.items()})
        if vid is None:
            return
        was = seg.get("speaker")
        if was is not None and was >= 0 and was != vid:      # it had been given to someone else: forget that
            try:
                n = self.voices.unlearn(pcm, was)
                if n:
                    print("[voices] dropped %d learned example(s) of %s that were really this voice" % (
                        n, "you" if was == self.voices.wearer_id else self.voices.names.get(was)))
            except Exception as e:  # noqa: BLE001
                print("[voices] could not unlearn:", e)
        seg["speaker"], seg["is_wearer"] = vid, vid == self.voices.wearer_id
        seg["name"] = self.voices.names.get(vid)
        self.screen.send("relabel", seg=seg_id, speaker=vid, is_wearer=seg["is_wearer"], name=seg["name"])
        print("[voices] line %s assigned to %s" % (seg_id, "you" if seg["is_wearer"] else seg["name"]))
        self.relabel_unknown(vid, seg)

    def relabel_unknown(self, vid, taught):
        """After a correction: look back over the lines still marked "Someone" (the last three
        minutes, while their audio is kept) and give this voice the ones that now match it.
        Lines too short to judge follow the line they are attached to."""
        segs, moved = list(self.segments), {taught["id"]}
        for s in segs:
            if s.get("speaker") != UNKNOWN or s["id"] in moved:
                continue
            pcm = self.audio.slice(s["a0"] - 0.05, s["a1"] + 0.05)
            if len(pcm) < 32000:
                continue
            try:
                who, lead = self.voices.identify_enrolled(pcm, loose=True)
            except Exception as e:  # noqa: BLE001
                print("[voices] look-back failed:", e)
                return
            if who == vid and lead >= 0.12:              # a smaller lead is enough once you have said they are here
                moved.add(s["id"])
        changed = True
        while changed:                                   # short fragments right next to a matched line
            changed = False
            for a, b in zip(segs, segs[1:]):
                for short, other in ((a, b), (b, a)):
                    if (short["id"] not in moved and other["id"] in moved and short.get("speaker") == UNKNOWN
                            and short["a1"] - short["a0"] < 1.0 and abs(b["a0"] - a["a1"]) < 0.6):
                        moved.add(short["id"])
                        changed = True
        n = 0
        for s in segs:
            if s["id"] in moved and s is not taught:
                if s["a1"] - s["a0"] >= 1.5:             # a line long enough to learn from: one more example of this voice
                    try:
                        self.voices.teach(self.audio.slice(s["a0"] - 0.05, s["a1"] + 0.05), vid=vid)
                    except Exception as e:  # noqa: BLE001
                        print("[voices] could not learn from a look-back line:", e)
                s["speaker"], s["is_wearer"] = vid, vid == self.voices.wearer_id
                s["name"] = self.voices.names.get(vid)
                self.screen.send("relabel", seg=s["id"], speaker=vid, is_wearer=s["is_wearer"], name=s["name"])
                n += 1
        if n:
            print("[voices] look-back: %d earlier line(s) also given to %s" % (n, "you" if vid == self.voices.wearer_id else self.voices.names.get(vid)))

    def find_voice(self, label):
        """'Person 2', 'speaker two', 'Ardek' -> that voice's id, or None."""
        want = " ".join(str(label or "").lower().split())
        if not want:
            return None
        digits = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6"}
        want = " ".join(digits.get(w, w) for w in want.replace("speaker", "person").split())
        for vid, name in self.voices.names.items():
            if name.lower() == want:
                return vid
        return None

    def learn_names(self, entry):
        try:
            addressed, self_name = brain.find_names(entry["text"])
        except Exception as e:  # noqa: BLE001
            print("[names] failed:", e)
            return
        vid = entry["speaker"]
        # only name a voice once Dextra knows which voice is the wearer, and never from a guess
        if (self_name and vid >= 0 and self.voices.wearer_id is not None and vid != self.voices.wearer_id
                and entry.get("voice_sim", 0) >= 0.42):
            self.set_name(vid, self_name, "said their own name")
        elif self_name and vid < 0 and self.voices.wearer_id is not None:
            self.pending_name = (self_name, time.time())    # their next longer sentence gets the name
        if addressed and vid == self.voices.wearer_id and addressed not in self.voices.names.values():
            self.pending_name = (addressed, time.time())
            print("[names] you addressed %s; the next new voice to answer gets that name" % addressed)

    def names_msg(self):
        return json.dumps({"kind": "names", "t": time.time(), "names": {str(k): v for k, v in self.voices.names.items()}})

    def _save_loop(self):
        while True:
            time.sleep(10)
            try:
                if not self.voices.enrolled:
                    self.voices.save(self.voices_path)
                elif getattr(self.band, "transport_kind", "") != "sim":   # a pretend band teaches nothing real
                    self.voices.save_learned(self.learned_path)
            except Exception as e:  # noqa: BLE001
                print("[voices] save failed:", e)

    def wearer_score(self, voice_id, rms, pose):
        """Wrist raised: loudness decides (15 dB gap in calibration), and confident sentences
        teach Dextra which voiceprint is the wearer. Arm down: the voiceprint decides."""
        loud = self.loud.wearer_score(rms, pose)
        if loud is not None:
            if loud >= 0.8:
                self.voices.vote_wearer(voice_id, True)
            elif loud <= 0.2:
                self.voices.vote_wearer(voice_id, False)
            if self.voices.wearer_id is None:
                return loud
            match = 1.0 if voice_id == self.voices.wearer_id else 0.0
            return round(0.6 * loud + 0.4 * match, 2)
        if self.voices.wearer_id is None:
            return 0.5
        if voice_id < 0:
            return 0.5                               # undecided voice
        return 0.9 if voice_id == self.voices.wearer_id else 0.1

    def act(self, mode, entry):
        aid = uuid.uuid4().hex[:8]
        if not entry.get("soft"):
            self.voice.chime()                       # heard at once, before Gemini has answered
            self.card(aid, "working", mode, entry["text"], result="Thinking...")
        self.pool.submit(self._act, mode, entry, aid)

    def _act(self, mode, entry, aid):
        try:
            if mode == "command":
                out = None
                if entry.get("parsed") is not None:
                    try:
                        out = entry["parsed"].result(timeout=10)      # already asked while you were finishing
                    except Exception:  # noqa: BLE001
                        out = None
                if out is None:
                    by_age = sorted(self.scheduler.pending(), key=lambda r: r.get("created", 0))
                    out = brain.interpret_command(entry["text"], [r["task"] for r in by_age])
                intent = out.get("intent")
                if entry.get("soft") and intent not in ("task", "question", "label_speaker", "cancel_reminder"):
                    print("[command] soft, not actionable, ignored:", entry["text"])
                    return
                if intent == "task" and out["trigger"]["type"] == "activity" and not self.activity_ok:
                    say = "I can't tell when you move on this band yet."
                    self.screen.send("action", id=aid, status="done", mode=mode, text=entry["text"], result=say)
                elif intent == "task":
                    r = self.scheduler.add_task(out.get("task"), out["trigger"], out["action"], out.get("say"))
                    when = spoken_time(r["remind_at"]) if r["kind_of"] == "time" else r["trigger_label"]
                    say = out.get("say") or "Okay, %s." % r["trigger_label"]
                    self.screen.send("reminder", **r)
                    self.screen.send("reminders", items=self.scheduler.pending())
                    self.screen.send("action", id=aid, status="done", mode=mode, text=entry["text"],
                                     result="Task: %s, %s" % (r["task"], when))
                elif intent == "question":
                    say = brain.answer_question(entry["text"], self.transcript_lines(), self.reminder_lines())
                    self.screen.send("action", id=aid, status="done", mode="ask", text=entry["text"], result=say)
                elif intent == "label_speaker":
                    others = [s for s in list(self.segments)
                              if s["speaker"] >= 0 and s["mode"] != "command"
                              and s["speaker"] not in (self.voices.wearer_id, entry["speaker"])]
                    name = (out.get("name") or "").strip().title()
                    if self.voices.wearer_id is None:
                        say = "I don't know your voice yet. Talk into the band first."
                    elif name and self.find_voice(out.get("who")) is not None:
                        vid = self.find_voice(out.get("who"))       # "Person 2 was Hardik"
                        old = self.voices.names.get(vid)
                        self.set_name(vid, name, "you told me")
                        say = "Got it, %s is %s." % (old, name)
                    elif others and name:
                        self.set_name(others[-1]["speaker"], name, "you told me")
                        say = "Got it, that's %s." % name
                    else:
                        say = "I haven't heard anyone else yet."
                    self.screen.send("action", id=aid, status="done", mode=mode, text=entry["text"], result=say)
                elif intent == "cancel_reminder":
                    chosen = out.get("cancel") if isinstance(out.get("cancel"), list) else []
                    chosen = [str(c) for c in chosen]
                    if [c.strip().lower() for c in chosen] == ["all"]:
                        gone = self.scheduler.cancel("all")
                    else:
                        gone = self.scheduler.cancel("", chosen) if chosen else []
                        if not gone:                 # nothing matched exactly: the newest one
                            gone = self.scheduler.cancel("")
                    if not gone:
                        say = "There's nothing to cancel."
                    elif len(gone) == 1:
                        say = "Cancelled %s." % gone[0]["task"]
                    else:
                        say = "Cancelled %d reminders: %s." % (len(gone), ", ".join(r["task"] for r in gone))
                    for r in gone:
                        self.screen.send("reminder", **r)
                    self.screen.send("reminders", items=self.scheduler.pending())
                    self.screen.send("action", id=aid, status="done", mode=mode, text=entry["text"], result=say)
                else:
                    say = out.get("say") or "Done."
                    self.screen.send("action", id=aid, status="done", mode=mode, text=entry["text"], result=say)
                self.voice.say(say)
            elif mode == "ask":
                say = brain.answer_question(entry["text"], self.transcript_lines(), self.reminder_lines())
                self.screen.send("action", id=aid, status="done", mode=mode, text=entry["text"], result=say)
                self.voice.say(say)
        except Exception as e:  # noqa: BLE001
            print("[act] failed:", e)
            self.screen.send("action", id=aid, status="refused", mode=mode, text=entry["text"], reason="Something went wrong")

    def transcript_lines(self):
        out = []
        for s in list(self.segments)[-60:]:
            who = ("You" if s["speaker"] == self.voices.wearer_id or s["wearer"] >= WEARER_MIN
                   else self.voices.names.get(s["speaker"]) or "Speaker %s" % s["speaker"])
            out.append("[%s] %s: %s" % (time.strftime("%-I:%M:%S", time.localtime(s["start"])), who, s["text"]))
        return out

    def reminder_lines(self):
        return ["%s, %s" % (r["task"], r.get("trigger_label") or spoken_time(r["remind_at"]))
                for r in self.scheduler.pending()]

    def reminders_msg(self):
        return json.dumps({"kind": "reminders", "t": time.time(), "items": self.scheduler.pending()})

    def on_reminder_due(self, r):
        """A task's moment came (called from the scheduler's thread, or a worker for event triggers)."""
        print("[task] due: %s (%s)" % (r["task"], r.get("trigger_label")))
        if (r.get("action") or {}).get("type") == "answer":
            self.pool.submit(self.run_answer_task, r)     # needs Gemini: keep the scheduler free
            return
        self.screen.send("reminder", **r)
        self.screen.send("reminders", items=self.scheduler.pending())
        r = dict(r, task=(r.get("action") or {}).get("text") or r["task"])   # what gets spoken
        acked = threading.Event()
        self.unacked.append((r, acked))

        def nag():                            # repeat until a thumbs up, three times at most
            for _ in range(3):
                self.voice.say("Reminder: %s." % r["task"])
                if acked.wait(12):
                    return
            self.unacked = [(x, e) for x, e in self.unacked if x is not r]
        threading.Thread(target=nag, daemon=True).start()

    def run_answer_task(self, r):
        """A task whose action is worked out when it fires: answer from the conversation so far."""
        aid = uuid.uuid4().hex[:8]
        title = "%s (%s)" % (r["task"], r.get("trigger_label") or "now")
        self.card(aid, "working", "ask", title, result="Thinking...")
        try:
            prompt = r["action"].get("prompt") or r["task"]
            if r.get("heard_line"):
                prompt += " (Just now %s said: \"%s\")" % (r.get("heard_by") or "someone", r["heard_line"])
            say = brain.answer_question(prompt, self.transcript_lines(), self.reminder_lines())
        except Exception as e:  # noqa: BLE001
            print("[task] answer failed:", e)
            say = "I couldn't work that out."
        self.card(aid, "done", "ask", title, result=say)
        self.screen.send("reminders", items=self.scheduler.pending())
        self.voice.say(say)

    def check_heard(self, entry):
        """Runs on a worker for each ambient line while any "when someone mentions..." task waits."""
        speaker = "You" if entry["is_wearer"] else (entry.get("name") or "Someone")
        for r in self.scheduler.waiting("heard"):
            trig = r["trigger"]
            who = (trig.get("who") or "anyone").strip()
            if who.lower() not in ("anyone", "someone", "anybody", ""):
                if speaker.lower() != who.lower() and not (who.lower() in ("i", "me", "you") and entry["is_wearer"]):
                    continue
            if not keyword_hit(entry["text"], trig.get("keywords")):
                continue
            ok = brain.heard_match(entry["text"], speaker, who, trig.get("about") or r["task"])
            print("[task] heard check %r for %r -> %s" % (entry["text"][:50], r["task"], ok))
            if ok:                                # None (Gemini unreachable) does not fire
                self.scheduler.fire(r["id"], heard_line=entry["text"], heard_by=speaker)

    def on_activity(self, e):
        """The band's own still / walk / run, once a second: drives "when I get up" tasks."""
        state = getattr(e, "state", None)
        if state is None:
            return
        moving = state in ("walk", "run")
        self.move_run = self.move_run + 1 if moving == self.move_last else 1
        self.move_last = moving
        if self.move_run == 2 and moving != self.moving:      # two readings in a row: a real change
            self.moving = moving
            print("[activity] %s" % ("moving" if moving else "still"))
            self.screen.send("activity", state=state, moving=moving)
            if self.scheduler.waiting("activity"):
                self.pool.submit(self.scheduler.fire_activity, "start_moving" if moving else "stop_moving")

    # ---- gestures

    def arm_swing(self, t0, t1):
        """How far the direction of gravity swung between two times, in degrees: the largest angle
        between any reading and the average. None when there are too few readings to say."""
        vs = [(x, y, z) for t, x, y, z in list(self.arm) if t0 <= t <= t1]
        if len(vs) < 4:
            return None
        mean = [sum(v[i] for v in vs) / len(vs) for i in range(3)]
        mn = math.sqrt(sum(c * c for c in mean))
        worst = 0.0
        for v in vs:
            n = math.sqrt(sum(c * c for c in v))
            if n < 1e-6 or mn < 1e-6:
                continue
            cos = sum(a * b for a, b in zip(v, mean)) / (n * mn)
            worst = max(worst, math.degrees(math.acos(max(-1.0, min(1.0, cos)))))
        return worst

    def on_gesture(self, label, prob):
        if label != "fist":
            return                                # thumbs up is out of the demo: taps acknowledge
        print("[gesture] %s (%.2f)" % (label, prob))
        self.screen.send("gesture", label=label, prob=prob)
        if label == "fist":
            now = time.time()                     # the window ended ~0.9 s ago; look just before it
            swing = self.arm_swing(now - 1.9, now - 1.3)
            if swing is not None and swing > FIST_SWING_DEG:
                print("[gesture] fist ignored: arm was moving (swung %.0f degrees)" % swing)
                return
            self.toggle_pause()
        elif label == "thumbs_up":
            self.acknowledge("Thumbs up")

    def acknowledge(self, how):
        waiting, self.unacked = self.unacked, []
        if not waiting:
            return
        for _, acked in waiting:
            acked.set()
        self.card(uuid.uuid4().hex[:8], "done", "command", how,
                  result="Acknowledged: %s" % ", ".join(r["task"] for r, _ in waiting))
        self.voice.say("Got it.")

    def toggle_pause(self):
        """Fist: the microphone really turns off (the band's blue light goes out)."""
        if time.time() - getattr(self, "last_toggle", 0) < 2.0:
            return                            # one toggle per gesture, however it was detected
        self.last_toggle = time.time()
        if not self.paused:
            self.paused, self.pause_started = True, time.time()
            self.band.configure(audio=False, mic_level=False)
            self.screen.send("mic", on=False)
            self.voice.say("Off the record.")
            self.last_toggle = time.time()
        else:
            self.new_decoder()                # the band numbers audio packets from 0 again
            self.band.configure(audio=True, mic_level=True)
            if self.audio_t0 is not None:     # Deepgram's clock only counts audio it received
                self.audio_t0 += time.time() - self.pause_started
            self.paused = False
            self.screen.send("mic", on=True)
            self.voice.say("Listening.")

    def status(self):
        pending = self.scheduler.pending()
        if not pending:
            self.voice.say("No reminders.")
            return
        first = pending[0]
        more = " and %d more" % (len(pending) - 1) if len(pending) > 1 else ""
        when = spoken_time(first["remind_at"]) if first["remind_at"] else first.get("trigger_label", "")
        self.voice.say("%s %s%s." % (first["task"], when, more))

    def clip(self):
        now = time.time()
        recent = [s for s in self.segments if s["end"] >= now - CLIP_SECONDS]
        if not recent:
            self.screen.send("clip", text="(nothing said in the last %d s)" % CLIP_SECONDS, speaker=None)
            return
        others = [s for s in recent if s["wearer"] < 0.5]
        chosen = others or recent
        speaker = chosen[-1]["speaker"]
        text = " ".join(s["text"] for s in chosen if s["speaker"] == speaker)
        self.screen.send("clip", text=text, speaker=speaker)
        print("[clip] speaker %s: %s" % (speaker, text))


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve_http():
    web = ROOT / "web" / "dist" if (ROOT / "web" / "dist" / "index.html").exists() else ROOT / "web"
    handler = functools.partial(QuietHandler, directory=str(web))
    http.server.ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), handler).serve_forever()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("number", nargs="?", type=int, help="band number on the bag")
    p.add_argument("--usb", action="store_true", help="band on the USB cable instead of Bluetooth")
    p.add_argument("--sim", action="store_true", help="pretend band with the kit's sample speech")
    args = p.parse_args()
    if not args.sim and args.number is None:
        p.error("give your band number, or --sim")
    load_env()

    screen, voice = Screen(), Voice()
    voice.prewarm(["Got it.", "Off the record.", "Listening.", "Only the wearer can do that.", "Done.",
                   "There's nothing to cancel.", "Sorry, I couldn't do that."])
    threading.Thread(target=serve_http, daemon=True).start()

    extra = {}
    if args.sim:
        import kit
        from vexo_open import audio
        extra["audio_packets"] = audio.ogg_packets(kit.SAMPLE_SPEECH)
    band = None
    for attempt in range(90):            # keep trying ~3 minutes: double-tap the band any time
        try:
            band = Band.connect(args.number or 42, "sim" if args.sim else ("serial" if args.usb else "ble"),
                                auto_reconnect=True, **extra)
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 0:
                print("Waiting for the band. Double-tap it (light blinks blue)...")
            time.sleep(2)
    if band is None:
        sys.exit("Could not connect to the band.")
    print("Connected to", band.name)

    cast = Dextra(band, screen, voice)
    if not cast.poses.calibrated:
        print("No calibration yet: every pose reads as 'moving'. Run the calibration page first (python -m cast.calibration).")
    cast.ears = ears.DiarizedDeepgram(on_segment=cast.on_segment, on_partial=cast.on_partial, send="pcm")
    cast.ears.start()

    band.on("motion", cast.on_motion)
    band.on("mic-level", cast.on_mic)
    band.on("audio", cast.on_audio)
    band.on("double-tap", cast.on_double_tap)
    band.on("tap", cast.on_tap)

    lost_at = [None]

    def on_reconnecting(e):
        lost_at[0] = lost_at[0] or time.time()
        print("[band] link lost, reconnecting...")

    def on_reconnected(e):
        cast.new_decoder()
        if cast.audio_t0 is not None and lost_at[0]:
            cast.audio_t0 += time.time() - lost_at[0]
        lost_at[0] = None
        print("[band] reconnected")
    band.on("reconnecting", on_reconnecting)
    band.on("reconnected", on_reconnected)
    try:
        band.set_tx_power(4)                 # strongest radio: the room is full of Bluetooth
    except Exception as e:  # noqa: BLE001
        print("tx power:", e)
    gain = float(os.environ.get("CAST_MIC_GAIN", "20"))
    if gain != 20:
        band.set_mic_gain(gain)
        print("Mic gain %+.0f dB (loudness readings shift by %+.0f dB)" % (gain, 20 - gain))
    cast.db_shift = 20 - gain
    if cast.fast:        # 100 readings a second of both sensors; 208 flooded the link next to the mic
        band.configure(motion=gestures.RATE, sensors="both", events=True, audio=True, mic_level=True)
    else:
        band.configure(motion=50, events=True, audio=True, mic_level=True)
    band.on("notice", cast.on_notice)

    def on_battery(m):
        try:
            screen.send("battery", percent=int(m.percent), charging=bool(getattr(m, "charging", False)))
        except Exception as e:  # noqa: BLE001
            print("[battery]", e)
    try:                                         # the band's battery level, for the top bar
        band.on("battery", on_battery)
        band.configure(battery=30)               # a reading every 30 seconds
        def battery_loop():                      # ask outright too: the stream only speaks when the level changes,
            while True:                          # and a browser opened later would never see the first reading
                try:
                    on_battery(band.get_battery())
                except Exception as e:  # noqa: BLE001
                    print("[battery] read failed:", str(e)[:80])
                    time.sleep(50)
                time.sleep(10)
        threading.Thread(target=battery_loop, daemon=True).start()
    except Exception as e:  # noqa: BLE001 - older firmware or the pretend band: just no battery shown
        print("[battery] not available:", str(e)[:100])
    try:                                         # still / walk / run from the band, once a second
        band.on("activity", cast.on_activity)
        band.configure(activity=True)
        cast.activity_ok = True
    except Exception as e:  # noqa: BLE001 - tasks on movement are optional
        print("Activity stream not available:", str(e)[:80])
    print("Microphone on (solid blue light). Tell people nearby. Screen: http://localhost:%d" % HTTP_PORT)

    async def run():
        screen.loop = asyncio.get_running_loop()
        async with websockets.serve(screen.handler, "127.0.0.1", WS_PORT):
            while band.connected or band.reconnecting:
                await asyncio.sleep(0.5)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cast.ears.stop(wait=2)
        except Exception:
            pass
        band.disconnect()


if __name__ == "__main__":
    main()
