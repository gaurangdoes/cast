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
import websockets  # noqa: E402

import calibrate  # noqa: E402  (pose list, talk tests, summarize)
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
    + [{"id": "cover:open", "group": "Covering the band", "title": "Open", "instruction": "Arm resting. YOU talk normally."},
       {"id": "cover:covered", "group": "Covering the band", "title": "Covered",
        "instruction": "Cover the band with your other hand and keep talking."}]
)


def load_results():
    try:
        r = json.loads(OUT.read_text())
        if r.get("source") == "calib_server":
            return r
    except (OSError, ValueError):
        pass
    return {"source": "calib_server", "poses": {}, "talk": {}, "cover": {}}


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
        self.rec.start()
        t0 = time.time()
        while time.time() - t0 < RECORD_S:
            self.active = {"id": step_id, "phase": "recording", "left": round(RECORD_S - (time.time() - t0), 1)}
            self.push_state()
            time.sleep(0.25)
        summary = calibrate.summarize(*self.rec.stop())
        kind, key = step_id.split(":", 1)
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
        return {"pairs": pairs[:6], "loudness": loud, "cover_drop": (c - o) if o is not None and c is not None else None}

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
        for kind, bucket in (("pose", "poses"), ("talk", "talk"), ("cover", "cover")):
            for k, v in self.results[bucket].items():
                done["%s:%s" % (kind, k)] = v
        return json.dumps({"kind": "cal_state", "band": "Vexo-HACK-%02d" % self.number, "status": self.status,
                           "error": self.error, "steps": STEPS, "done": done, "active": self.active,
                           "analysis": self.analysis()})

    def live_msg(self):
        g = None
        if self.g:
            n = math.sqrt(sum(x * x for x in self.g)) or 1
            g = [round(x / n, 3) for x in self.g]
        return json.dumps({"kind": "cal_live", "gravity": g, "gyro": round(self.gyro, 1), "rms_db": self.rms,
                           "nearest": self.nearest_pose()})

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
