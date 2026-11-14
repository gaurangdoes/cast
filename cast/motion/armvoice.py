"""Can the band feel the wearer's own voice as vibration through the arm?

A test, not a feature. Three 15 s recordings are made from the calibration page in the real
operating condition (microphone on, accelerometer at 416 readings a second): nobody talking,
the wearer talking with the arm resting, and someone else talking loudly nearby. If the
wearer's voice shakes the wrist, the accelerometer shows extra energy between 70 and 200 Hz
(where the pitch of a voice sits) only in the second recording.

    .venv/bin/python -m cast.motion.armvoice        # prints the same analysis the page shows
"""
import numpy as np

from ..paths import DATA

FOLDER = DATA / "arm_voice"
LABELS = ("silent", "wearer", "other")
BAND_HZ = (70.0, 200.0)
WINDOW_S = 0.5


def band_energy(t_ms, acc):
    """Vibration between 70 and 200 Hz, in thousandths of g (RMS), for each half-second window.
    Uses the real spacing of the samples: the band's "416 a second" is a few percent off."""
    t = np.asarray(t_ms, dtype=float)
    acc = np.asarray(acc, dtype=float)
    if len(t) < 50:
        return np.array([]), 0.0
    rate = (len(t) - 1) / max(1e-6, (t[-1] - t[0]) / 1000.0)
    mag = np.linalg.norm(acc[:, :3], axis=1)
    n = int(WINDOW_S * rate)
    out = []
    for a in range(0, len(mag) - n + 1, n):
        w = mag[a:a + n]
        w = (w - w.mean()) * np.hanning(n)
        spec = np.fft.rfft(w)
        freqs = np.fft.rfftfreq(n, 1.0 / rate)
        sel = (freqs >= BAND_HZ[0]) & (freqs <= min(BAND_HZ[1], rate / 2 - 1))
        # RMS of the in-band part of the signal (the Hann window halves the power on average: undo it)
        power = 2.0 * np.sum(np.abs(spec[sel]) ** 2) / (n * n) / 0.375
        out.append(np.sqrt(power) * 1000.0)
    return np.array(out), rate


def analyse(folder=None):
    """Summary for the page and the script. Uses the newest recording of each kind."""
    folder = folder or FOLDER
    result = {"recordings": {}, "verdict": None}
    energy = {}
    for label in LABELS:
        files = sorted(folder.glob(label + "_*.npz"))
        if not files:
            continue
        d = np.load(files[-1])
        e, rate = band_energy(d["t"], d["acc"])
        if len(e) == 0:
            continue
        energy[label] = e
        result["recordings"][label] = {
            "windows": int(len(e)), "rate": round(float(rate)),
            "median_mg": round(float(np.median(e)), 3),
            "p10_mg": round(float(np.percentile(e, 10)), 3),
            "p90_mg": round(float(np.percentile(e, 90)), 3),
        }
    if all(l in energy for l in LABELS):
        line = max(float(np.percentile(energy["silent"], 95)), float(np.percentile(energy["other"], 95)))
        share = float((energy["wearer"] > line).mean())
        if share > 0.8:
            word, text = "usable", "The wearer's voice clearly shakes the wrist."
        elif share >= 0.4:
            word, text = "marginal", "There is some signal, but too much overlap to rely on by itself."
        else:
            word, text = "absent", "The wearer's voice does not show up above the other two recordings."
        result["verdict"] = {"line_mg": round(line, 3), "share_above": round(share, 2), "word": word, "text": text}
    return result


def describe(result):
    lines = []
    for label in LABELS:
        r = result["recordings"].get(label)
        if r:
            lines.append("%-7s median %.3f mg   10th-90th percentile %.3f - %.3f mg   (%d windows at %d/s)" % (
                label, r["median_mg"], r["p10_mg"], r["p90_mg"], r["windows"], r["rate"]))
        else:
            lines.append("%-7s not recorded yet" % label)
    v = result["verdict"]
    if v:
        lines.append("%d%% of the wearer's half-seconds are above %.3f mg, the level 95%% of the silent and "
                     "other-person half-seconds stay under." % (round(v["share_above"] * 100), v["line_mg"]))
        lines.append("Verdict: %s. %s" % (v["word"].upper(), v["text"]))
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe(analyse()))
