"""Wrist pose classifier and wearer score, built from data/calibration.json.

Pose: the direction of gravity in the band's frame (smoothed accelerometer), matched to the
nearest calibrated pose. A pose only counts once it has held for HOLD_S, so the mode does
not flicker while the arm moves.

Wearer score: how likely a stretch of speech came from the person wearing the band, from
how loud it was at the wrist compared with the calibrated "you" and "teammate" loudness
for the pose the arm was in.
"""
import json
import math
import time
from collections import deque
from ..paths import DATA

CALIBRATION = DATA / "calibration.json"

MAX_ANGLE_DEG = 28      # further than this from every calibrated pose: "moving"
HOLD_S = 0.3            # a pose must hold this long to count
SMOOTH = 0.15           # low-pass on acceleration (0..1, smaller = smoother)

MODE_OF_POSE = {
    "mouth": "command",
    "ear": "command",   # dropped as its own gesture: raised wrist = talk to Dextra
    "chest": "ambient",
    "extended": "ambient",
    "glance": "ambient",     # dropped: only the raised wrist does anything
    "rest_table": "ambient",
    "side": "ambient",
}


def _unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _angle(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))))


class PoseTracker:
    def __init__(self, calibration=None):
        cal = calibration if calibration is not None else load_calibration()
        # only the three poses in use; older entries in the file were recorded with the band
        # sitting differently on the wrist
        self.refs = {name: _unit(p["gravity"]) for name, p in (cal.get("poses") or {}).items()
                     if p.get("gravity") and name in ("mouth", "rest_table", "side")}
        self.g = None
        self.candidate, self.candidate_since = "moving", 0.0
        self.pose = "moving"
        self.history = deque(maxlen=6000)        # (wall time, stable pose), 20 a second: 5 minutes
        self._last_hist = 0.0
        self.ghist = deque(maxlen=2400)          # (wall time, gravity direction), 20 a second: 2 minutes
        self._last_g = 0.0
        self.last_angle = None

    @property
    def calibrated(self):
        return bool(self.refs)

    def feed(self, ax, ay, az, now=None):
        now = now or time.time()
        if self.g is None:
            self.g = [ax, ay, az]
        else:
            self.g = [g + SMOOTH * (v - g) for g, v in zip(self.g, (ax, ay, az))]
        label, best = "moving", None
        if self.refs:
            u = _unit(self.g)
            best = min(((_angle(u, ref), name) for name, ref in self.refs.items()))
            self.last_angle = best[0]
            if best[0] <= MAX_ANGLE_DEG:
                label = best[1]
        if label != self.candidate:
            self.candidate, self.candidate_since = label, now
        if now - self.candidate_since >= HOLD_S:
            self.pose = self.candidate
        if now - self._last_hist >= 0.05:
            self._last_hist = now
            self.history.append((now, self.pose))
        if now - self._last_g >= 0.05:
            self._last_g = now
            self.ghist.append((now, tuple(self.g)))
        return self.pose

    def angle_to(self, name, t_start, t_end):
        """Degrees between a pose's reference and how the wrist was held between two times."""
        ref = getattr(self, "refs", {}).get(name)
        gs = [g for t, g in getattr(self, "ghist", ()) if t_start - 0.2 <= t <= t_end + 0.2]
        if ref is None or not gs:
            return None
        return round(_angle(ref, _unit([sum(g[i] for g in gs) / len(gs) for i in range(3)])), 1)

    def adapt(self, name, t_start, t_end, weight=0.3):
        """Nudge a pose's reference toward how the wrist was actually held between two times.
        Called when something else proves the pose (speech loud enough that the band must be at
        the mouth), so the poses follow the band if it shifts on the wrist."""
        gs = [g for t, g in self.ghist if t_start <= t <= t_end]
        if name not in self.refs or len(gs) < 10:
            return None
        mean = [sum(g[i] for g in gs) / len(gs) for i in range(3)]
        spread = max(max(abs(g[i] - mean[i]) for g in gs) for i in range(3))
        if spread > 0.35:                          # the arm was moving: not a clean example
            return None
        new = _unit(mean)
        moved = _angle(self.refs[name], new)
        others = [_angle(ref, new) for n, ref in self.refs.items() if n != name]
        if moved > 25 or (others and min(others) < moved):
            return None                            # too far off, or closer to another pose: not this pose
        self.refs[name] = _unit([(1 - weight) * a + weight * b for a, b in zip(self.refs[name], new)])
        try:                                       # keep it for the next start
            cal = json.loads(CALIBRATION.read_text())
            cal["poses"][name]["gravity"] = [round(x, 3) for x in self.refs[name]]
            CALIBRATION.write_text(json.dumps(cal, indent=2))
        except Exception:  # noqa: BLE001
            pass
        return moved

    def pose_during(self, t_start, t_end):
        """Most common stable pose between two wall-clock times (falls back to now)."""
        counts = {}
        for t, p in self.history:
            if t_start - 0.3 <= t <= t_end + 0.3:
                counts[p] = counts.get(p, 0) + 1
        if not counts:
            return self.pose
        # a raised pose anywhere in the utterance wins over resting/moving
        for raised in ("mouth", "ear"):
            if counts.get(raised, 0) >= 0.3 * sum(counts.values()):
                return raised
        return max(counts, key=counts.get)


class Loudness:
    """Timeline of the band's mic level (rms dB below full scale: smaller = louder)."""

    def __init__(self, calibration=None):
        cal = calibration if calibration is not None else load_calibration()
        self.timeline = deque(maxlen=3000)       # (wall time, rms_db), 10/s
        self.thresholds = {}
        talk = cal.get("talk") or {}
        for pose in ("mouth", "rest_table", "side"):
            you = talk.get(pose + "/you", {}).get("rms_db_median")
            mate = talk.get(pose + "/teammate", {}).get("rms_db_median")
            if you is not None and mate is not None and mate - you >= 6:
                self.thresholds[pose] = ((you + mate) / 2.0, max(2.0, (mate - you) / 4.0))

    def feed(self, rms_db, now=None):
        self.timeline.append((now or time.time(), rms_db))

    def level_during(self, t_start, t_end):
        vals = sorted(v for t, v in self.timeline if t_start <= t <= t_end)
        if not vals:
            return None
        return vals[max(0, len(vals) // 4)]      # a loud-ish reading: 25th percentile of dB

    def wearer_score(self, rms_db, pose):
        """0..1 from loudness, or None when loudness can't tell for this pose.
        Calibration showed a clear gap only with the wrist raised (mouth/ear)."""
        if rms_db is None:
            return None
        key = "mouth" if pose in ("mouth", "ear") else pose
        if key not in self.thresholds:
            return None
        mid, width = self.thresholds[key]
        return round(1.0 / (1.0 + math.exp((rms_db - mid) / width)), 2)


def load_calibration():
    try:
        return json.loads(CALIBRATION.read_text())
    except (OSError, ValueError):
        return {}


class LearnedPoses(PoseTracker):
    """Same interface as PoseTracker, but poses come from the model trained in the lab
    (data/gestures/pose_model.pkl) on 20 s varied recordings of each pose."""
    NAME = {"raised": "mouth", "rest": "rest_table", "side": "side"}

    def __init__(self):
        from . import gestures
        self.live = gestures.LivePose()
        self.pose = "moving"
        self.history = deque(maxlen=4000)        # one entry per 50 ms: over 3 minutes
        self.last_angle = None
        self._last_hist = 0.0

    @property
    def calibrated(self):
        return True

    def feed(self, ax, ay, az, t_ms=None, now=None):
        now = now or time.time()
        self.pose = self.NAME.get(self.live.feed(t_ms if t_ms is not None else now * 1000, ax, ay, az), "moving")
        if now - self._last_hist >= 0.05:
            self._last_hist = now
            self.history.append((now, self.pose))
        return self.pose
