"""Calibration with a screen. Run this, then open http://localhost:8000/#calibrate

    .venv/bin/python calib_server.py 45

The page walks you through each step with a Record button and shows live readings from
the band. Every recorded step is saved to data/calibration.json straight away, so you can
redo any single step. Close this (Ctrl+C) before starting app.py: one program per band.
"""
import argparse
import asyncio
import functools
import http.server
import json
import math
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vexo-open-kit" / "python"))
import numpy as np  # noqa: E402
import websockets  # noqa: E402

import calibrate  # noqa: E402  (pose list, talk tests, summarize)
import gestures  # noqa: E402
import subprocess  # noqa: E402
from vexo_open.sync import Band  # noqa: E402

OUT = ROOT / "data" / "calibration.json"
PREP_S, RECORD_S = 3, 4

STEPS = (
    [{"id": "pose:" + n, "group": "Poses (stay quiet)", "title": n.replace("_", " ").title(), "instruction": t}
     for n, t in calibrate.POSES]
    + [{"id": "talk:%s/%s" % (p, w), "group": "Who is talking",
        "title": "%s · %s" % (p.replace("_", " ").title(), "you talk" if w == "you" else "teammate talks"),
        "instruction": "%s. %s." % (dict(calibrate.POSES)[p],
                                    "YOU talk normally" if w == "you" else "Stay quiet; a TEAMMATE talks from about 1 m away")}
       for p, w in calibrate.TALK_TESTS]
    + [{"id": "probe:rest", "group": "Fist test (fast motion)", "title": "Still hand",
        "instruction": "Forearm on the table, hand open and relaxed. Don't move."},
       {"id": "probe:fist", "group": "Fist test (fast motion)", "title": "Clench and release",
        "instruction": "Forearm on the table. Clench your fist hard, release, about once a second."},
       {"id": "probe:hold", "group": "Fist test (fast motion)", "title": "Fist closed",
        "instruction": "Forearm on the table. Close your fist and keep it closed."},
       {"id": "probe:move", "group": "Fist test (fast motion)", "title": "Normal movement",
        "instruction": "Move your arm around and gesture like you're talking. Keep your hand open."}]
)


def load_results():
    try:
        r = json.loads(OUT.read_text())
        if r.get("source") == "calib_server":
            r.setdefault("probe", {})
            return r
    except (OSError, ValueError):
        pass
    return {"source": "calib_server", "poses": {}, "talk": {}, "cover": {}, "probe": {}}


def probe_summary(samples):
    """High-frequency vibration energy (what a tendon twitch would add) in 100 ms windows."""
    if len(samples) < 200:
        return {"samples": len(samples)}
    a = np.array(samples, dtype=np.float64)
    t, acc = a[:, 0], a[:, 1:4]
    rate = (len(t) - 1) / max(1e-6, (t[-1] - t[0]) / 1000.0)
    k = 15                                                     # ~18 ms moving average at 833/s
    kernel = np.ones(k) / k
    smooth = np.stack([np.convolve(acc[:, i], kernel, mode="same") for i in range(3)], axis=1)
    hp = np.linalg.norm(acc - smooth, axis=1)[k:-k]
    win = max(1, int(rate / 10))
    energy = np.array([np.sqrt(np.mean(hp[i:i + win] ** 2)) for i in range(0, len(hp) - win, win)]) * 1000
    return {"samples": len(samples), "rate": round(rate), "hp_median_mg": round(float(np.median(energy)), 2),
            "hp_p95_mg": round(float(np.percentile(energy, 95)), 2), "hp_max_mg": round(float(energy.max()), 2)}


class Calib:
    def __init__(self, number, transport):
        self.number, self.transport = number, transport
        self.band = None
        self.status, self.error = "idle", None
        self.rec = calibrate.Recorder()
        self.g = None
        self.gyro = 0.0
        self.rms = None
        self.results = load_results()
        self.active = None            # {"id", "phase", "left"}
        self.loop = None
        self.clients = set()
        self.lab = {"recording": None, "report": self._load_report(), "live": False, "detections": {}, "last": None,
                    "pose_report": self._load_report("pose_report.json"), "pose": None, "pose_raw": None}
        self.lab_buf = None              # fast samples while a lab recording runs
        self.detector = None

    # ---- gesture lab

    @staticmethod
    def _load_report(name="report.json"):
        try:
            return json.loads((gestures.DATA / name).read_text())
        except (OSError, ValueError):
            return None

    def say(self, text):
        subprocess.Popen(["say", "-r", "210", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def lab_record(self, label, reps):
        if self.status != "connected" or self.lab["recording"] or self.active:
            return
        threading.Thread(target=self._lab_record, args=(label, reps), daemon=True).start()

    def _lab_record(self, label, reps):
        name = {"fist": "clench your fist twice, quickly", "thumbs_up": "thumbs up, then back",
                "none": "move normally"}.get(label, label)
        pose_say = {"pose_raised": "Hold your wrist up near your mouth, like talking into it",
                    "pose_rest": "Rest your arm on the table or your lap",
                    "pose_side": "Let your arm hang at your side",
                    "pose_point": "Point your arm at someone, palm down"}
        self.band.configure(motion=gestures.RATE, sensors="both")
        self.lab_buf = []
        cues = []
        try:
            if label == "none" or label.startswith("pose_"):
                self.lab["recording"] = {"label": label, "rep": 0, "reps": 0, "phase": "go", "left": reps}
                self.push_state()
                if label == "none":
                    self.say("Move normally for %d seconds. Talk with your hands, scratch, rest. Don't do the gestures." % reps)
                else:
                    self.say("%s. Keep changing it a little: different angles, sit, stand." % pose_say.get(label, label))
                    time.sleep(2.5)
                for left in range(reps, 0, -1):
                    self.lab["recording"]["left"] = left
                    self.push_state()
                    time.sleep(1)
            else:
                self.lab["recording"] = {"label": label, "rep": 0, "reps": reps, "phase": "ready"}
                self.push_state()
                self.say("Get ready. When I say now, %s. Move your arm to a new position every few reps." % name)
                time.sleep(6.5)
                for i in range(reps):
                    self.lab["recording"] = {"label": label, "rep": i + 1, "reps": reps, "phase": "now"}
                    self.push_state()
                    if self.lab_buf:
                        cues.append((self.lab_buf[-1][0] - self.lab_buf[0][0]) / 1000.0 + 0.25)  # + speech delay
                    self.say("now")
                    time.sleep(0.8)
                    self.lab["recording"]["phase"] = "wait"
                    self.push_state()
                    time.sleep(1.8)
                self.say("Done.")
            buf = self.lab_buf
            if buf and len(buf) > gestures.RATE:
                a = np.array(buf)
                gestures.Recordings.save(label, a[:, 0], a[:, 1:7], cues)
        finally:
            self.lab_buf = None
            self.lab["recording"] = None
            self.band.configure(motion=50, sensors="both")
            self.push_state()

    def lab_train(self):
        self.lab["report"] = {"training": True}
        self.push_state()
        try:
            self.lab["report"] = gestures.train()
        except Exception as e:  # noqa: BLE001
            self.lab["report"] = {"error": str(e)}
        try:
            self.lab["pose_report"] = gestures.train_poses()
        except Exception as e:  # noqa: BLE001
            self.lab["pose_report"] = {"error": str(e)}
        self.push_state()

    def lab_live(self, on):
        if on:
            try:
                self.posemodel = gestures.LivePose() if gestures.POSE_MODEL.exists() else None
            except Exception:  # noqa: BLE001
                self.posemodel = None
            try:
                self.detector = gestures.LiveDetector()
            except Exception as e:  # noqa: BLE001
                self.lab["report"] = dict(self.lab.get("report") or {}, error="Train first (%s)" % e)
                self.push_state()
                return
            self.band.configure(motion=gestures.RATE, sensors="both")
            self.lab.update(live=True, detections={}, last=None)
        else:
            self.lab["live"] = False
            self.detector = None
            self.posemodel = None
            self.band.configure(motion=50, sensors="both")
        self.push_state()

    # ---- voice lab: clean samples of each person through the band's own microphone

    def voice_record(self, label, seconds):
        if self.status != "connected" or self.lab["recording"] or self.active:
            return
        threading.Thread(target=self._voice_record, args=(label, seconds), daemon=True).start()

    def _voice_record(self, label, seconds):
        from vexo_open.audio import PcmDecoder
        dec, chunks = PcmDecoder(), []
        handler = lambda a: chunks.append(dec.feed(a))
        try:
            self.lab["recording"] = {"label": "voice_" + label, "rep": 0, "reps": 0, "phase": "ready", "left": seconds}
            self.push_state()
            who = "The wearer talks" if label == "wearer" else "The other person talks, the wearer stays silent"
            self.say("%s for %d seconds. Keep talking the whole time." % (who, seconds))
            time.sleep(4.5)
            self.band.on("audio", handler)
            self.band.configure(audio=True)
            for left in range(seconds, 0, -1):
                self.lab["recording"] = {"label": "voice_" + label, "rep": 0, "reps": 0, "phase": "go", "left": left}
                self.push_state()
                time.sleep(1)
            self.band.configure(audio=False)
            out = ROOT / "data" / "voices_lab"
            out.mkdir(parents=True, exist_ok=True)
            pcm = b"".join(chunks)
            (out / ("%s_%d.pcm" % (label, int(time.time())))).write_bytes(pcm)
            self.say("Done.")
        finally:
            self.lab["recording"] = None
            self.push_state()

    def lab_state(self):
        vc = {}
        for f in (ROOT / "data" / "voices_lab").glob("*.pcm"):
            k = f.stem.rsplit("_", 1)[0]
            vc[k] = vc.get(k, 0) + 1
        return dict(self.lab, counts=gestures.Recordings.counts(), voice_counts=vc)

    # ---- band

    def connect(self):
        if self.status in ("connecting", "connected"):
            return
        self.status, self.error = "connecting", None
        self.push_state()
        try:
            band = Band.connect(self.number, self.transport)
            band.on("motion", self.on_motion)
            band.on("mic-level", self.on_mic)
            band.on("disconnect", lambda e: self.set_status("disconnected", "The band disconnected."))
            band.configure(motion=50, mic_level=True)
            self.band = band
            self.set_status("connected")
        except Exception as e:  # noqa: BLE001
            self.set_status("error", str(e))

    def set_status(self, status, error=None):
        self.status, self.error = status, error
        self.push_state()

    def on_motion(self, m):
        if self.lab_buf is not None and m.gx is not None:
            self.lab_buf.append((m.t, m.ax, m.ay, m.az, m.gx, m.gy, m.gz))
        if getattr(self, "posemodel", None) is not None:
            before = self.lab.get("pose")
            pose = self.posemodel.feed(m.t, m.ax, m.ay, m.az)
            self.lab["pose"] = pose
            self.lab["pose_raw"] = self.posemodel.raw
            if pose != before:
                self.push_state()
        if self.detector is not None:
            hit = self.detector.feed(m.t, m.ax, m.ay, m.az, m.gx, m.gy, m.gz) if m.gx is not None else None
            if hit:
                label, prob = hit
                self.lab["detections"][label] = self.lab["detections"].get(label, 0) + 1
                self.lab["last"] = {"label": label, "prob": prob, "at": time.time()}
                self.say(label.replace("_", " "))
                self.push_state()
        if getattr(self, "fast", None) is not None and self.rec.on:
            self.fast.append((m.t, m.ax, m.ay, m.az))
            return
        if m.gx is None:                      # accelerometer-only fast mode
            v = (m.ax, m.ay, m.az)
            self.g = list(v) if self.g is None else [g + 0.04 * (x - g) for g, x in zip(self.g, v)]
            return
        self.rec.on_motion(m)
        v = (m.ax, m.ay, m.az)
        self.g = list(v) if self.g is None else [g + 0.15 * (x - g) for g, x in zip(self.g, v)]
        self.gyro = math.sqrt(m.gx ** 2 + m.gy ** 2 + m.gz ** 2)

    def on_mic(self, e):
        self.rec.on_mic(e)
        if getattr(e, "rms_db", None) is not None:
            self.rms = e.rms_db

    # ---- recording

    def record(self, step_id):
        if self.status != "connected" or self.active:
            return
        threading.Thread(target=self._record, args=(step_id,), daemon=True).start()

    def _record(self, step_id):
        for left in range(PREP_S, 0, -1):
            self.active = {"id": step_id, "phase": "countdown", "left": left}
            self.push_state()
            time.sleep(1)
        fast = step_id.startswith("probe:")
        if fast:
            self.band.configure(motion=833, sensors="accel")
            self.fast = []
        self.rec.start()
        t0 = time.time()
        while time.time() - t0 < RECORD_S:
            self.active = {"id": step_id, "phase": "recording", "left": round(RECORD_S - (time.time() - t0), 1)}
            self.push_state()
            time.sleep(0.25)
        motion, mic = self.rec.stop()
        kind, key = step_id.split(":", 1)
        if fast:
            self.band.configure(motion=50, sensors="both")
            summary = probe_summary(self.fast)
            np.save(ROOT / "data" / ("probe_%s.npy" % key), np.array(self.fast, dtype=np.float64))
            self.fast = None
            self.results.setdefault("probe", {})[key] = summary
            OUT.write_text(json.dumps(self.results, indent=2))
            self.active = None
            self.push_state()
            return
        summary = calibrate.summarize(motion, mic)
        self.results[{"pose": "poses", "talk": "talk", "cover": "cover"}[kind]][key] = summary
        self.results["recorded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.results["band"] = getattr(self.band, "name", None)
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(self.results, indent=2))
        self.active = None
        self.push_state()

    def reset(self):
        self.results = {"source": "calib_server", "poses": {}, "talk": {}, "cover": {}}
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(self.results, indent=2))
        self.push_state()

    # ---- what the page sees

    def analysis(self):
        poses = {n: p["gravity"] for n, p in self.results["poses"].items() if p.get("gravity")}
        names = sorted(poses)
        pairs = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pairs.append({"a": a, "b": b, "deg": round(calibrate.angle_deg(poses[a], poses[b]), 1)})
        pairs.sort(key=lambda x: x["deg"])
        loud = []
        for pose in ("mouth", "rest_table", "side"):
            you = self.results["talk"].get(pose + "/you", {}).get("rms_db_median")
            mate = self.results["talk"].get(pose + "/teammate", {}).get("rms_db_median")
            loud.append({"pose": pose, "you": you, "teammate": mate,
                         "gap": (mate - you) if you is not None and mate is not None else None})
        o = self.results["cover"].get("open", {}).get("rms_db_median")
        c = self.results["cover"].get("covered", {}).get("rms_db_median")
        return {"pairs": pairs[:6], "loudness": loud, "cover_drop": (c - o) if o is not None and c is not None else None,
                "probe": self.results.get("probe", {})}

    def nearest_pose(self):
        if not self.g:
            return None
        n = math.sqrt(sum(x * x for x in self.g)) or 1
        u = [x / n for x in self.g]
        best = None
        for name, p in self.results["poses"].items():
            if p.get("gravity"):
                d = calibrate.angle_deg(u, p["gravity"])
                if best is None or d < best[1]:
                    best = (name, d)
        return {"pose": best[0], "deg": round(best[1], 1)} if best else None

    def state_msg(self):
        done = {}
        for kind, bucket in (("pose", "poses"), ("talk", "talk"), ("cover", "cover"), ("probe", "probe")):
            for k, v in self.results[bucket].items():
                done["%s:%s" % (kind, k)] = v
        return json.dumps({"kind": "cal_state", "band": "Vexo-HACK-%02d" % self.number, "status": self.status,
                           "error": self.error, "steps": STEPS, "done": done, "active": self.active,
                           "analysis": self.analysis(), "lab": self.lab_state()})

    def live_msg(self):
        g = None
        if self.g:
            n = math.sqrt(sum(x * x for x in self.g)) or 1
            g = [round(x / n, 3) for x in self.g]
        return json.dumps({"kind": "cal_live", "gravity": g, "gyro": round(self.gyro, 1), "rms_db": self.rms,
                           "nearest": self.nearest_pose(), "pose": self.lab.get("pose"),
                           "pose_raw": self.lab.get("pose_raw")})

    def push_state(self):
        if self.loop and not self.loop.is_closed():
            msg = self.state_msg()
            try:
                self.loop.call_soon_threadsafe(self._broadcast, msg)
            except RuntimeError:
                pass

    def _broadcast(self, msg):
        for ws in list(self.clients):
            asyncio.ensure_future(self._send(ws, msg))

    async def _send(self, ws, msg):
        try:
            await ws.send(msg)
        except Exception:  # noqa: BLE001
            self.clients.discard(ws)

    async def handler(self, ws):
        self.clients.add(ws)
        await ws.send(self.state_msg())
        try:
            async for raw in ws:
                try:
                    cmd = json.loads(raw)
                except ValueError:
                    continue
                if cmd.get("cmd") == "connect":
                    threading.Thread(target=self.connect, daemon=True).start()
                elif cmd.get("cmd") == "record":
                    self.record(str(cmd.get("step")))
                elif cmd.get("cmd") == "reset":
                    self.reset()
                elif cmd.get("cmd") == "lab_record":
                    self.lab_record(str(cmd.get("label")), int(cmd.get("reps", 10)))
                elif cmd.get("cmd") == "voice_record":
                    self.voice_record(str(cmd.get("label")), int(cmd.get("seconds", 40)))
                elif cmd.get("cmd") == "lab_train":
                    threading.Thread(target=self.lab_train, daemon=True).start()
                elif cmd.get("cmd") == "lab_live":
                    threading.Thread(target=self.lab_live, args=(bool(cmd.get("on")),), daemon=True).start()
                elif cmd.get("cmd") == "lab_reset_counts":
                    self.lab.update(detections={}, last=None)
                    self.push_state()
        finally:
            self.clients.discard(ws)

    async def live_loop(self):
        while True:
            if self.status == "connected":
                self._broadcast(self.live_msg())
            await asyncio.sleep(0.1)


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve_http():
    web = ROOT / "web" / "dist"
    handler = functools.partial(Quiet, directory=str(web))
    http.server.ThreadingHTTPServer(("127.0.0.1", 8000), handler).serve_forever()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("number", nargs="?", type=int, help="band number on the bag")
    p.add_argument("--sim", action="store_true", help="pretend band")
    args = p.parse_args()
    if not args.sim and args.number is None:
        p.error("give your band number, or --sim")
    cal = Calib(args.number or 42, "sim" if args.sim else "ble")
    threading.Thread(target=serve_http, daemon=True).start()
    print("Open http://localhost:8000/#calibrate  (Ctrl+C to stop)")

    async def run():
        cal.loop = asyncio.get_running_loop()
        async with websockets.serve(cal.handler, "127.0.0.1", 8765):
            await cal.live_loop()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    finally:
        if cal.band:
            cal.band.disconnect()


if __name__ == "__main__":
    main()
