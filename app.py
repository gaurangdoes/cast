"""Cast: the main program. One process owns the band.

    .venv/bin/python app.py 42          # your band number
    .venv/bin/python app.py --sim       # pretend band; plays the kit's sample speech

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
import subprocess
import sys
import threading
import uuid
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vexo-open-kit" / "python"))
sys.path.insert(0, str(ROOT / "vexo-open-kit" / "examples" / "python"))

import websockets  # noqa: E402

import brain  # noqa: E402
import ears  # noqa: E402
import nemotron  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from scheduler import Scheduler, spoken_time  # noqa: E402
import gestures  # noqa: E402
from poses import MODE_OF_POSE, LearnedPoses, Loudness, PoseTracker, load_calibration  # noqa: E402
from vexo_open.sync import Band  # noqa: E402

HTTP_PORT = 8000
WS_PORT = 8765
WEARER_MIN = 0.6
REFUSE_BELOW = 0.35     # only refuse when fairly sure it's someone else
COMMAND_SETTLE_S = 1.2  # wait this long after the last raised-wrist words before acting
CLIP_SECONDS = 30
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
            async for _ in ws:
                pass
        finally:
            self.clients.discard(ws)


class Voice:
    """Spoken replies. macOS `say` for now; ElevenLabs later. While it talks, speech the
    band hears is ignored so Cast does not transcribe itself."""

    def __init__(self):
        self.busy_until = 0.0

    def say(self, text):
        self.busy_until = time.time() + 0.5 + len(text) * 0.07
        subprocess.Popen(["say", "-r", "205", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
        print("Loading the voice model...")
        from vexo_open.audio import PcmDecoder
        from voices import AudioBuffer, Voices
        self.voices = Voices()               # SpeechBrain ECAPA voiceprints (CPU)
        self.audio = AudioBuffer()
        self.decoder = PcmDecoder()
        self.last_voice = None
        self.pending_cmd = None               # raised-wrist speech collected until the speaker pauses
        self.last_partial = 0.0
        self.cmd_lock = threading.Lock()
        threading.Thread(target=self._command_loop, daemon=True).start()
        screen.snapshot = lambda: [self.reminders_msg(), json.dumps({"kind": "mic", "t": time.time(), "on": not self.paused})]

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
        self.ears.feed(a)
        try:
            self.audio.add(self.decoder.feed(a))
        except Exception:  # noqa: BLE001 - a bad packet must not stop transcription
            pass

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
        local_to_voice = {}
        for spk in {t[0] for t in turns}:
            audio = b"".join(self.audio.slice(a, b) for s2, a, b in turns if s2 == spk)
            vid, sim = self.voices.identify(audio)
            if vid is not None:
                local_to_voice[spk] = (vid, sim)

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
            vid, sim = local_to_voice.get(r["local"], (None, 0.0))
            if vid is None:                                  # no turn found: fingerprint the words themselves
                vid, sim = self.voices.identify(self.audio.slice(start - 0.05, end + 0.05))
            if vid is None:
                vid = self.last_voice if self.last_voice is not None else 0
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
        vid, sim = run["voice"], run["sim"]
        score = self.wearer_score(vid, rms, pose)
        mode = MODE_OF_POSE.get(pose, "ambient")
        entry = {"speaker": vid, "text": run["text"], "start": start, "end": end,
                 "pose": pose, "rms_db": rms, "wearer": score, "mode": mode,
                 "is_wearer": vid == self.voices.wearer_id, "voice_sim": round(sim, 2)}
        self.segments.append(entry)
        self.screen.send("segment", **entry)
        print("[voice %s%s sim %.2f | %-10s | wearer %.2f | %s] %s" % (
            vid, "*" if entry["is_wearer"] else "", sim, pose, score, mode, run["text"]))
        if mode in ("command", "ask"):
            with self.cmd_lock:                      # collect until the speaker pauses
                if self.pending_cmd is None:
                    self.pending_cmd = {"texts": [], "scores": [], "entry": entry}
                self.pending_cmd["texts"].append(entry["text"])
                self.pending_cmd["scores"].append(score)
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
            entry = dict(p["entry"], text=text, wearer=score)
            if score < REFUSE_BELOW:
                self.card(uuid.uuid4().hex[:8], "refused", "command", text,
                          reason="Not the wearer's voice (score %.2f)" % score)
                self.voice.say("Only the wearer can do that.")
            else:
                self.act("command", entry)

    def card(self, aid, status, mode, text, result=None, reason=None):
        self.screen.send("action", id=aid, status=status, mode=mode, text=text, result=result, reason=reason)

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
        return 0.9 if voice_id == self.voices.wearer_id else 0.1

    def act(self, mode, entry):
        aid = uuid.uuid4().hex[:8]
        self.card(aid, "working", mode, entry["text"], result="Thinking...")
        self.pool.submit(self._act, mode, entry, aid)

    def _act(self, mode, entry, aid):
        try:
            if mode == "command":
                out = brain.interpret_command(entry["text"])
                intent = out.get("intent")
                if intent == "reminder":
                    r = self.scheduler.add(out.get("task"), out["remind_at_epoch"], out.get("say"))
                    say = out.get("say") or "Okay, I'll remind you %s." % spoken_time(r["remind_at"])
                    self.screen.send("reminder", **r)
                    self.screen.send("reminders", items=self.scheduler.pending())
                    self.screen.send("action", id=aid, status="done", mode=mode, text=entry["text"],
                                     result="Reminder: %s %s" % (r["task"], spoken_time(r["remind_at"])))
                elif intent == "question":
                    say = brain.answer_question(entry["text"], self.transcript_lines(), self.reminder_lines())
                    self.screen.send("action", id=aid, status="done", mode="ask", text=entry["text"], result=say)
                elif intent == "cancel_reminder":
                    r = self.scheduler.cancel_latest()
                    say = "Cancelled %s." % r["task"] if r else "There's nothing to cancel."
                    if r:
                        self.screen.send("reminder", status="cancelled", **r)
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
            who = "You" if s["wearer"] >= WEARER_MIN else "Speaker %s" % s["speaker"]
            out.append("[%s] %s: %s" % (time.strftime("%-I:%M:%S", time.localtime(s["start"])), who, s["text"]))
        return out

    def reminder_lines(self):
        return ["%s at %s" % (r["task"], time.strftime("%-I:%M %p", time.localtime(r["remind_at"])))
                for r in self.scheduler.pending()]

    def reminders_msg(self):
        return json.dumps({"kind": "reminders", "t": time.time(), "items": self.scheduler.pending()})

    def on_reminder_due(self, r):
        print("[reminder] due:", r["task"])
        self.screen.send("reminder", **r)
        self.screen.send("reminders", items=self.scheduler.pending())
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

    # ---- gestures

    def on_gesture(self, label, prob):
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
        self.voice.say("%s %s%s." % (first["task"], spoken_time(first["remind_at"]), more))

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
            band = Band.connect(args.number or 42, "sim" if args.sim else ("serial" if args.usb else "ble"), **extra)
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
        print("No calibration yet: every pose reads as 'moving'. Run calibrate.py first.")
    cast.ears = ears.DiarizedDeepgram(on_segment=cast.on_segment, on_partial=cast.on_partial)
    cast.ears.start()

    band.on("motion", cast.on_motion)
    band.on("mic-level", cast.on_mic)
    band.on("audio", cast.on_audio)
    band.on("double-tap", cast.on_double_tap)
    band.on("tap", cast.on_tap)
    if cast.fast:        # 208 readings a second of both sensors: the most Bluetooth carries next to the mic
        band.configure(motion=gestures.RATE, sensors="both", events=True, audio=True, mic_level=True)
    else:
        band.configure(motion=50, events=True, audio=True, mic_level=True)
    band.on("notice", lambda n: print("[band notice]", n if isinstance(n, dict) else getattr(n, "__dict__", n), "| streams now:", band.streams))
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
