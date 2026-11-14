"""Calibration recorder for Cast.

Walks you through each wrist pose and a few talking tests, records what the band
sees, and saves the numbers to data/calibration.json. The pose classifier and the
wearer score are built from this file.

    .venv/bin/python calibrate.py 42        # your band number (on the bag)
    .venv/bin/python calibrate.py --sim     # pretend band, to check the script runs

Close the kit's dashboard tab first: the band takes one connection at a time.
Double-tap the band right before running (its light blinks blue).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vexo-open-kit" / "python"))
from vexo_open.sync import Band  # noqa: E402

POSES = [
    ("mouth", "Raise the band to your mouth, as if talking into it"),
    ("ear", "Raise the band to your ear, as if on a phone call"),
    ("rest_table", "Rest your forearm on the table"),
    ("side", "Let your arm hang at your side"),
    ("chest", "Hold the band against your chest, forearm across your body"),
    ("extended", "Hold your arm out straight in front of you, palm down"),
    ("glance", "Look at the band like checking a watch"),
]

TALK_TESTS = [
    ("mouth", "you"), ("mouth", "teammate"),
    ("rest_table", "you"), ("rest_table", "teammate"),
    ("side", "you"), ("side", "teammate"),
]

PREP_S = int(__import__("os").environ.get("PREP_S", 3))
RECORD_S = int(__import__("os").environ.get("RECORD_S", 4))


class Recorder:
    def __init__(self):
        self.motion = []
        self.mic = []
        self.on = False

    def on_motion(self, m):
        if self.on:
            self.motion.append((m.ax, m.ay, m.az, m.gx, m.gy, m.gz))

    def on_mic(self, e):
        if self.on:
            rms = getattr(e, "rms_db", None)
            if rms is not None:
                self.mic.append(rms)

    def start(self):
        self.motion, self.mic, self.on = [], [], True

    def stop(self):
        self.on = False
        return list(self.motion), list(self.mic)


def summarize(motion, mic):
    out = {"samples": len(motion), "mic_readings": len(mic)}
    if motion:
        n = len(motion)
        ax = sum(s[0] for s in motion) / n
        ay = sum(s[1] for s in motion) / n
        az = sum(s[2] for s in motion) / n
        norm = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
        out["gravity"] = [round(ax / norm, 3), round(ay / norm, 3), round(az / norm, 3)]
        # how still the arm was: spread of acceleration around its mean
        spread = math.sqrt(sum((s[0] - ax) ** 2 + (s[1] - ay) ** 2 + (s[2] - az) ** 2 for s in motion) / n)
        out["accel_spread_g"] = round(spread, 4)
        out["gyro_mean_dps"] = round(sum(math.sqrt(s[3] ** 2 + s[4] ** 2 + s[5] ** 2) for s in motion) / n, 1)
    if mic:
        mic_sorted = sorted(mic)
        out["rms_db_median"] = mic_sorted[len(mic_sorted) // 2]
        out["rms_db_min"] = mic_sorted[0]          # smaller = louder
    return out


def countdown(text, seconds):
    print("\n>>> " + text)
    for s in range(seconds, 0, -1):
        print("    starting in %d..." % s, end="\r", flush=True)
        time.sleep(1)
    print("    RECORDING (%ds) — hold still / keep going" % RECORD_S + " " * 10)


def record(band, rec, text):
    countdown(text, PREP_S)
    rec.start()
    band.wait(RECORD_S)
    return summarize(*rec.stop())


def angle_deg(a, b):
    dot = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))
    return math.degrees(math.acos(dot))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("number", nargs="?", type=int, help="band number on the bag, e.g. 42")
    p.add_argument("--usb", action="store_true", help="band on the USB cable instead of Bluetooth")
    p.add_argument("--sim", action="store_true", help="use the kit's pretend band")
    p.add_argument("--skip-talk", action="store_true", help="only record poses")
    args = p.parse_args()
    if not args.sim and args.number is None:
        p.error("give your band number (on the bag), or --sim")

    number = args.number or 42
    band = Band.connect(number, "sim" if args.sim else ("serial" if args.usb else "ble"))
    print("Connected to", band.name)

    rec = Recorder()
    band.on("motion", rec.on_motion)
    band.on("mic-level", rec.on_mic)
    band.configure(motion=50, mic_level=True)   # mic on: the band's light goes solid blue
    print("Microphone is on (solid blue light). Tell the people near you.")

    results = {"band": band.name, "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"), "poses": {}, "talk": {}, "cover": {}}

    print("\n=== Part 1: poses (stay quiet) ===")
    for name, text in POSES:
        results["poses"][name] = record(band, rec, "%s: %s" % (name.upper(), text))
        print("    gravity", results["poses"][name].get("gravity"))

    if not args.skip_talk:
        print("\n=== Part 2: who is talking ===")
        for pose, who in TALK_TESTS:
            say = "YOU talk normally" if who == "you" else "Stay quiet; your TEAMMATE talks from about 1 m away"
            pose_text = dict(POSES)[pose]
            results["talk"]["%s/%s" % (pose, who)] = record(band, rec, "%s. %s." % (pose_text, say))

        print("\n=== Part 3: covering the band ===")
        results["cover"]["open"] = record(band, rec, "Arm resting. YOU talk normally")
        results["cover"]["covered"] = record(band, rec, "Cover the band with your other hand and keep talking")

    band.disconnect()

    out = ROOT / "data" / "calibration.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("\nSaved", out)

    # Summary: how far apart the poses are. Under ~20 degrees means they may get confused.
    names = [n for n, _ in POSES if "gravity" in results["poses"][n]]
    print("\nClosest pose pairs (degrees between gravity directions):")
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairs.append((angle_deg(results["poses"][a]["gravity"], results["poses"][b]["gravity"]), a, b))
    for d, a, b in sorted(pairs)[:5]:
        flag = "  <-- may be confused" if d < 20 else ""
        print("  %-11s vs %-11s %5.1f°%s" % (a, b, d, flag))

    if results["talk"]:
        print("\nLoudness (rms dB below full scale; SMALLER = LOUDER):")
        for pose in ("mouth", "rest_table", "side"):
            you = results["talk"].get(pose + "/you", {}).get("rms_db_median")
            mate = results["talk"].get(pose + "/teammate", {}).get("rms_db_median")
            gap = (mate - you) if (you is not None and mate is not None) else None
            print("  %-11s you %-5s teammate %-5s gap %s dB" % (pose, you, mate, gap))
        o = results["cover"].get("open", {}).get("rms_db_median")
        c = results["cover"].get("covered", {}).get("rms_db_median")
        print("  covered: open %s -> covered %s (drop %s dB)" % (o, c, (c - o) if o is not None and c is not None else None))


if __name__ == "__main__":
    main()
