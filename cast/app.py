"""Cast: the main program. One process owns the band.

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
COMMAND_SETTLE_S = 1.8  # wait this long after the last raised-wrist words before acting
CLIP_SECONDS = 30
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
    falling back to the Mac's built-in voice if that fails or is slow. While Cast talks, speech
    the band hears is ignored so it does not transcribe itself. Replies play one at a time."""
    MODEL = os.environ.get("CAST_TTS_MODEL", "gemini-3.8-flash-lite-tts")
    NAME = os.environ.get("CAST_TTS_VOICE", "Kore")

    def __init__(self):
        self.busy_until = 0.0
        self.q = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, text):
        self.busy_until = max(self.busy_until, time.time() + 2.5 + len(text) * 0.07)   # until we know better
        self.q.put(text)

    def _run(self):
        while True:
            text = self.q.get()
            try:
                wav = self._gemini(text) if os.environ.get("CAST_TTS", "gemini") == "gemini" else None
            except Exception as e:  # noqa: BLE001
                print("[voice] Gemini speech failed, using the Mac voice:", str(e)[:100])
                wav = None
            try:
                if wav:
                    path, seconds = wav
                    self.busy_until = time.time() + seconds + 0.6
                    subprocess.run(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    self.busy_until = time.time() + 0.5 + len(text) * 0.07
                    subprocess.run(["say", "-r", "205", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:  # noqa: BLE001
                print("[voice] playback failed:", e)
            self.busy_until = max(self.busy_until, time.time() + 0.4)

    def _gemini(self, text):
        import base64
        import tempfile
        import urllib.request
        import wave
        body = {"contents": [{"parts": [{"text": text}]}],
                "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self.NAME}}}}}
        req = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent" % self.MODEL,
            data=json.dumps(body).encode(),
            headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"})
        part = json.load(urllib.request.urlopen(req, timeout=5))["candidates"][0]["content"]["parts"][0]["inlineData"]
        data = base64.b64decode(part["data"])
        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        if "wav" in part.get("mimeType", ""):
            Path(path).write_bytes(data)
        else:                                            # raw 16-bit samples at 24 kHz
            with wave.open(path, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(data)
        with wave.open(path, "rb") as w:
            seconds = w.getnframes() / float(w.getframerate())
        return path, seconds

    def speaking(self, t_start, t_end):
        return t_start <= self.busy_until


class Cast:
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
        self.unacked = None                   # (reminder, threading.Event) waiting for a thumbs up
        self.loud = Loudness(cal)
        self.segments = deque(maxlen=500)
        self.audio_t0 = None
        self.last_pose_sent = None
        self.last_motion_push = 0.0
        self.pool = ThreadPoolExecutor(max_workers=3)
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
        self.last_voice = None
        self.last_run_end = None
        self.turn = None                      # the stretch of continuous speech being labelled
        self.last_seq = None
        self.pending_cmd = None               # raised-wrist speech collected until the speaker pauses
        self.last_partial = 0.0
        self.cmd_lock = threading.Lock()
        threading.Thread(target=self._command_loop, daemon=True).start()
        screen.on_message = self.on_screen_message
        screen.snapshot = lambda: [self.reminders_msg(), self.names_msg(), json.dumps({"kind": "mic", "t": time.time(), "on": not self.paused})]

    # ---- band events (run on the band's handler thread; keep short)

    def on_motion(self, m):
        now = time.time()
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
        """A finished stretch of speech from Deepgram. Nemotron splits it into turns (who spoke
        when, even mid-sentence); each turn's voiceprint gives it a stable speaker id."""
        t0 = self.audio_t0 or time.time()
        if self.voice.speaking(t0 + words[0]["start"], t0 + words[-1]["end"]):
            return                                   # Cast's own spoken reply
        for run in self.split_by_speaker(words):
            self.handle_run(run)

    def split_by_speaker(self, words):
        w0, w1 = words[0]["start"], words[-1]["end"]
        win_start = max(0.0, w0 - CONTEXT_S)                 # extra audio before helps Nemotron
        pcm = self.audio.slice(win_start, w1 + 0.15)
        turns = []
        if len(pcm) > 16000:                                 # at least 0.5 s
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
        out = []
        for r in runs:
            start, end = r["words"][0]["start"], r["words"][-1]["end"]
            # Nemotron only decides where the speaker changes; who it is comes from this
            # stretch's own audio, never from the context before it
            vid, sim = self.voices.identify(self.audio.slice(start - 0.05, end + 0.05))
            # Someone still talking (under 0.7 s since their last words, and Nemotron saw no
            # change of speaker) is judged on everything they have said in this turn: longer
            # audio is what makes a voiceprint reliable (81% right at 1 s, 95% at 3 s).
            t = self.turn
            continuing = (t is not None and r is runs[0] and start - t["end"] < 0.7)
            changed = vid is not None and vid >= 0 and t is not None and vid != t["vid"] and sim >= 0.30
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
            if vid == UNKNOWN and end - self.turn["start"] >= 5.0:
                # five seconds of someone nobody recognises: a new person joins the conversation
                t0 = self.audio_t0 or time.time()
                level = self.loud.level_during(t0 + self.turn["start"], t0 + end)
                if level is None or level - getattr(self, "db_shift", 0) > RAISED_DB + 3:   # not the wearer up close
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
            out.append({"voice": vid, "sim": sim, "start": start, "end": end,
                        "text": " ".join(w["text"] for w in r["words"])})
        return out

    def handle_run(self, run):
        t0 = self.audio_t0 or time.time()
        start, end = t0 + run["start"], t0 + run["end"]
        if self.voice.speaking(start, end):
            return                                   # that was Cast's own voice
        pose = self.poses.pose_during(start, end)
        rms = self.loud.level_during(start, end)
        if rms is not None:
            rms -= getattr(self, "db_shift", 0)      # as if the mic were at the gain we calibrated with
        vid, sim = run["voice"], run["sim"]
        # Who is being talked to? Three signals, measured on this band: loudness at the mic
        # (12 dB at the mouth, 21 resting, 27 someone else), the wrist pose, and whose voice it is.
        loud = rms is not None and rms <= RAISED_DB                 # anyone speaking right into the band
        if vid < 0 and loud and self.voices.wearer_id is not None:
            vid = self.voices.wearer_id       # undecided voice, but this loud at the band is the wearer
        mine = vid == self.voices.wearer_id or vid < 0           # the wearer, or undecided
        near = rms is not None and rms <= NEAR_DB and pose == "mouth" and mine   # softer, wrist raised
        with self.cmd_lock:
            ongoing = self.pending_cmd is not None and mine         # still finishing a slow command
        # the wearer's own voice, fairly loud, but the pose is not clearly raised: treat it as a
        # command only if it turns out to be something Cast can do (decided after Gemini reads it)
        soft = (not loud and not near and not ongoing and vid == self.voices.wearer_id
                and rms is not None and rms <= SOFT_DB)
        if (rms is not None and rms <= RAISED_DB - 2 and vid == self.voices.wearer_id and end - start >= 1.0
                and hasattr(self.poses, "adapt")):
            # the wearer, this loud: the band is at the mouth for certain. Let the "raised" pose
            # follow where the wrist really was, so a shifted band corrects itself.
            moved = self.poses.adapt("mouth", start, end)
            if moved is not None and moved > 8:
                print("[pose] raised-wrist reference moved %.0f degrees toward how you hold it now" % moved)
        if loud or near or ongoing or soft:
            pose = "mouth"
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
                 "id": uuid.uuid4().hex[:8], "a0": run["start"], "a1": run["end"]}
        if mode == "ambient" and NAME_HINT.search(run["text"]):
            self.pool.submit(self.learn_names, entry)
        self.segments.append(entry)
        self.screen.send("segment", **entry)
        if mode == "ambient" and self.scheduler.waiting("heard"):
            self.pool.submit(self.check_heard, entry)
        print("[voice %s%s sim %.2f | %-10s | wearer %.2f | %s] %s" % (
            vid, "*" if entry["is_wearer"] else "", sim, pose, score, mode, run["text"]))
        if mode in ("command", "ask"):
            with self.cmd_lock:                      # collect until the speaker pauses
                if self.pending_cmd is None:
                    self.pending_cmd = {"texts": [], "scores": [], "entry": entry, "other": [], "soft": soft}
                self.pending_cmd["texts"].append(entry["text"])
                self.pending_cmd["scores"].append(score)
                # a recognised voice that is not the wearer's
                self.pending_cmd["other"].append(
                    self.voices.wearer_id is not None and vid >= 0 and vid != self.voices.wearer_id)
                self.pending_cmd["last"] = time.time()

    def _command_loop(self):
        while True:
            time.sleep(0.2)
            with self.cmd_lock:
                p = self.pending_cmd
                if p is None:
                    continue
                now = time.time()
                if now - p["last"] < COMMAND_SETTLE_S or now - self.last_partial < 0.8:
                    continue                         # still talking
                self.pending_cmd = None
            text = " ".join(p["texts"]).strip()
            score = max(p["scores"])
            if len(text.split()) < 3:
                print("[command] ignored fragment:", text)
                continue
            entry = dict(p["entry"], text=text, wearer=score, soft=p.get("soft", False))
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

    def on_screen_message(self, msg):
        """Typed on the screen: click a speaker's name and rename them."""
        if msg.get("cmd") == "assign":
            self.pool.submit(self.assign, str(msg.get("seg")), msg.get("name"))
        elif msg.get("cmd") == "rename":
            try:
                vid = int(msg.get("speaker"))
            except (TypeError, ValueError):
                return
            name = " ".join(str(msg.get("name") or "").split())[:30]
            if vid >= 0 and vid != self.voices.wearer_id and name:
                self.pool.submit(self.set_name, vid, name, "you typed it")

    def assign(self, seg_id, who):
        """Typed on the screen for a line Cast could not place: "that was me" / "Hardik" / a new
        name. The line is relabelled and its audio becomes an example of that voice, so Cast
        recognises them next time."""
        seg = next((s for s in list(self.segments) if s.get("id") == seg_id), None)
        who = " ".join(str(who or "").split())[:30]
        if seg is None or not who:
            return
        pcm = self.audio.slice(seg["a0"] - 0.05, seg["a1"] + 0.05)      # empty if older than 3 minutes
        if who.lower() in ("me", "you", "i", "myself", "mine"):
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
        seg["speaker"], seg["is_wearer"] = vid, vid == self.voices.wearer_id
        seg["name"] = self.voices.names.get(vid)
        self.screen.send("relabel", seg=seg_id, speaker=vid, is_wearer=seg["is_wearer"], name=seg["name"])
        print("[voices] line %s assigned to %s" % (seg_id, "you" if seg["is_wearer"] else seg["name"]))

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
        # only name a voice once Cast knows which voice is the wearer, and never from a guess
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
                else:
                    self.voices.save_learned(self.learned_path)
            except Exception as e:  # noqa: BLE001
                print("[voices] save failed:", e)

    def wearer_score(self, voice_id, rms, pose):
        """Wrist raised: loudness decides (15 dB gap in calibration), and confident sentences
        teach Cast which voiceprint is the wearer. Arm down: the voiceprint decides."""
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
            self.card(aid, "working", mode, entry["text"], result="Thinking...")
        self.pool.submit(self._act, mode, entry, aid)

    def _act(self, mode, entry, aid):
        try:
            if mode == "command":
                out = brain.interpret_command(entry["text"], [r["task"] for r in self.scheduler.pending()])
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
                    gone = self.scheduler.cancel("", [str(c) for c in chosen]) if chosen else self.scheduler.cancel("")
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
        self.unacked = (r, acked)

        def nag():                            # repeat until a thumbs up, three times at most
            for _ in range(3):
                self.voice.say("Reminder: %s." % r["task"])
                if acked.wait(12):
                    return
            if self.unacked and self.unacked[0] is r:
                self.unacked = None
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

    def on_gesture(self, label, prob):
        if label != "fist":
            return                                # thumbs up is out of the demo: taps acknowledge
        print("[gesture] %s (%.2f)" % (label, prob))
        self.screen.send("gesture", label=label, prob=prob)
        if label == "fist":
            now = time.time()                     # the window ended ~0.9 s ago; look just before it
            if self.poses.pose_during(now - 1.9, now - 1.3) == "moving":
                print("[gesture] fist ignored: arm was moving")
                return
            self.toggle_pause()
        elif label == "thumbs_up":
            self.acknowledge("Thumbs up")

    def acknowledge(self, how):
        if not self.unacked:
            return
        r, acked = self.unacked
        self.unacked = None
        acked.set()
        self.card(uuid.uuid4().hex[:8], "done", "command", how, result="Acknowledged: %s" % r["task"])
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

    cast = Cast(band, screen, voice)
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
