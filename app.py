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
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vexo-open-kit" / "python"))
sys.path.insert(0, str(ROOT / "vexo-open-kit" / "examples" / "python"))

import websockets  # noqa: E402

import brain  # noqa: E402
import ears  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from scheduler import Scheduler, spoken_time  # noqa: E402
from poses import MODE_OF_POSE, Loudness, PoseTracker, load_calibration  # noqa: E402
from vexo_open.sync import Band  # noqa: E402

HTTP_PORT = 8000
WS_PORT = 8765
WEARER_MIN = 0.6
CLIP_SECONDS = 30


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
        self.poses = PoseTracker(cal)
        self.loud = Loudness(cal)
        self.segments = deque(maxlen=500)
        self.audio_t0 = None
        self.last_pose_sent = None
        self.last_motion_push = 0.0
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.scheduler = Scheduler(self.on_reminder_due)
        self.last_glance_status = 0.0
        self.speaker_votes = {}         # diarized speaker id -> (+ wearer / - not wearer) evidence
        screen.snapshot = lambda: [self.reminders_msg()]

    # ---- band events (run on the band's handler thread; keep short)

    def on_motion(self, m):
        pose = self.poses.feed(m.ax, m.ay, m.az)
        now = time.time()
        if pose == "glance" and self.last_pose_sent != "glance" and now - self.last_glance_status > 8:
            self.last_glance_status = now
            self.pool.submit(self.status)
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

    def on_double_tap(self, e):
        self.clip()

    # ---- speech

    def on_partial(self, text):
        self.screen.send("partial", text=text)

    def on_segment(self, seg):
        t0 = self.audio_t0 or time.time()
        start, end = t0 + seg["start"], t0 + seg["end"]
        if self.voice.speaking(start, end):
            return                                   # that was Cast's own voice
        pose = self.poses.pose_during(start, end)
        rms = self.loud.level_during(start, end)
        score = self.wearer_score(seg["speaker"], rms, pose)
        mode = MODE_OF_POSE.get(pose, "ambient")
        entry = {"speaker": seg["speaker"], "text": seg["text"], "start": start, "end": end,
                 "pose": pose, "rms_db": rms, "wearer": score, "mode": mode}
        self.segments.append(entry)
        self.screen.send("segment", **entry)
        print("[spk %s | %-10s | wearer %.2f | %s] %s" % (seg["speaker"], pose, score, mode, seg["text"]))
        if mode in ("command", "ask"):
            if score >= WEARER_MIN:
                self.act(mode, entry)
            else:
                self.screen.send("action", status="refused", mode=mode, text=entry["text"],
                                 reason="Not the wearer (score %.2f)" % score)
                self.voice.say("Only the wearer can do that.")

    def wearer_score(self, speaker, rms, pose):
        """Loudness decides when the wrist is raised (clear 15 dB gap in calibration).
        Each confident raised-wrist sentence teaches Cast which speaker label is the wearer,
        so it keeps recognising them with the arm down, where loudness can't tell."""
        loud = self.loud.wearer_score(rms, pose)
        if loud is not None:
            if loud >= 0.8:
                self.speaker_votes[speaker] = self.speaker_votes.get(speaker, 0) + 1
            elif loud <= 0.2:
                self.speaker_votes[speaker] = self.speaker_votes.get(speaker, 0) - 1
            known = self.speaker_votes.get(speaker, 0)
            # blend: a known wearer voice nudges up, a known other voice nudges down
            return round(min(1.0, max(0.0, loud + 0.1 * max(-2, min(2, known)))), 2)
        votes = self.speaker_votes.get(speaker, 0)
        if votes > 0:
            return 0.85
        if votes < 0 or any(v > 0 for v in self.speaker_votes.values()):
            return 0.15                 # someone else is already known to be the wearer
        return 0.5

    def act(self, mode, entry):
        self.screen.send("action", status="working", mode=mode, text=entry["text"], result="Thinking...")
        self.pool.submit(self._act, mode, entry)

    def _act(self, mode, entry):
        try:
            if mode == "command":
                out = brain.interpret_command(entry["text"])
                intent = out.get("intent")
                if intent == "reminder":
                    r = self.scheduler.add(out.get("task"), out["remind_at_epoch"], out.get("say"))
                    say = out.get("say") or "Okay, I'll remind you %s." % spoken_time(r["remind_at"])
                    self.screen.send("reminder", status="scheduled", **r)
                    self.screen.send("reminders", items=self.scheduler.pending())
                    self.screen.send("action", status="done", mode=mode, text=entry["text"],
                                     result="Reminder: %s %s" % (r["task"], spoken_time(r["remind_at"])))
                elif intent == "cancel_reminder":
                    r = self.scheduler.cancel_latest()
                    say = "Cancelled %s." % r["task"] if r else "There's nothing to cancel."
                    if r:
                        self.screen.send("reminder", status="cancelled", **r)
                    self.screen.send("reminders", items=self.scheduler.pending())
                    self.screen.send("action", status="done", mode=mode, text=entry["text"], result=say)
                else:
                    say = out.get("say") or "Done."
                    self.screen.send("action", status="done", mode=mode, text=entry["text"], result=say)
                self.voice.say(say)
            elif mode == "ask":
                say = brain.answer_question(entry["text"], self.transcript_lines(), self.reminder_lines())
                self.screen.send("action", status="done", mode=mode, text=entry["text"], result=say)
                self.voice.say(say)
        except Exception as e:  # noqa: BLE001
            print("[act] failed:", e)
            self.screen.send("action", status="refused", mode=mode, text=entry["text"], reason="Something went wrong")

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
        self.voice.say("Reminder: %s." % r["task"])

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
    band = Band.connect(args.number or 42, "sim" if args.sim else ("serial" if args.usb else "ble"), **extra)
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
    band.configure(motion=50, events=True, audio=True, mic_level=True)
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
