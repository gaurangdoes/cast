"""Gesture detection from the band's accelerometer (416 readings/s, the rate the live app
can use alongside the microphone).

Pipeline:
  1. activity(): how much is happening right now, from fast vibration (a fist clench tightens
     wrist tendons) and from slower orientation change (a thumbs-up rolls the wrist).
  2. Around each activity peak, a 1.2 s window is cut out and turned into ~30 numbers
     (orientation before/at peak/after, how much and how fast it changed, vibration in
     three frequency bands).
  3. A random forest trained on your own recorded reps (plus "none": ordinary movement)
     says which gesture it was, with a probability.
"""
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "gestures"
MODEL = DATA / "model.pkl"
RATE = 416
PRE_S, POST_S = 0.35, 0.85            # window around an activity peak
HP_K = 9                               # moving-average length for the high-pass (~22 ms)
LP_K = 41                              # ~0.1 s smoothing for orientation


def _movavg(x, k):
    kernel = np.ones(k) / k
    return np.stack([np.convolve(x[:, i], kernel, mode="same") for i in range(x.shape[1])], axis=1)


def activity(acc):
    """Per-sample activity: fast vibration energy + orientation change speed (both in g-ish)."""
    lp = _movavg(acc, LP_K)
    hp = np.linalg.norm(acc - _movavg(acc, HP_K), axis=1)
    vib = np.sqrt(_movavg((hp ** 2)[:, None], HP_K * 3)[:, 0])
    turn = np.zeros(len(acc))
    step = int(0.1 * RATE)
    if len(lp) > step:
        turn[step:] = np.linalg.norm(lp[step:] - lp[:-step], axis=1)
    return vib * 8.0 + turn                 # vibration is small in g; weight it up


def _unit(v):
    return v / (np.linalg.norm(v) or 1.0)


def features(acc):
    """acc: (n, 3) window in g at ~RATE. Returns a 1-D feature vector."""
    n = len(acc)
    lp = _movavg(acc, LP_K)
    edge = max(4, int(0.12 * RATE))
    g0, g1 = _unit(lp[:edge].mean(0)), _unit(lp[-edge:].mean(0))
    dev = np.linalg.norm(lp - lp[:edge].mean(0), axis=1)
    pk = int(np.argmax(dev))
    gp = _unit(lp[max(0, pk - 5):pk + 5].mean(0))
    ang = lambda a, b: float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))
    hp = acc - _movavg(acc, HP_K)
    hpm = np.linalg.norm(hp, axis=1)
    spec = np.abs(np.fft.rfft(hpm - hpm.mean())) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / RATE)
    bands = [spec[(freqs >= a) & (freqs < b)].sum() for a, b in ((8, 30), (30, 70), (70, 200))]
    tot = sum(bands) or 1.0
    act = activity(acc)[LP_K:-LP_K]          # skip the window edges (smoothing artefacts)
    above = float((act > 0.5 * act.max()).mean()) if len(act) and act.max() > 0 else 0.0
    # No absolute orientation here on purpose: with it the model learned "the arm position I
    # recorded in" instead of the gesture, and fired every second in that position.
    return np.concatenate([
        gp - g0, g1 - g0,                                    # how gravity moved, relative to the start
        [ang(g0, gp), ang(gp, g1), ang(g0, g1)],             # how far it turned, and whether it came back
        lp.std(0),                                           # slow movement per axis
        [hpm.mean() * 1000, np.percentile(hpm, 95) * 1000, hpm.max() * 1000],   # vibration (mg)
        [b / tot for b in bands], [np.log1p(tot)],
        [pk / n, above, act.max() if len(act) else 0.0],
    ])


def peak_windows(t_ms, acc, cue_times=None, min_gap_s=0.9, thresh=None):
    """Cut gesture windows. With cue_times (seconds, band clock): one window per cue, at the
    strongest activity within [cue-0.2, cue+1.6]. Without: every activity peak above thresh."""
    t = (np.asarray(t_ms) - t_ms[0]) / 1000.0
    act = activity(acc)
    pre, post = int(PRE_S * RATE), int(POST_S * RATE)
    out = []
    if cue_times is not None:
        for c in cue_times:
            idx = np.nonzero((t >= c - 0.2) & (t <= c + 1.6))[0]
            if len(idx) == 0:
                continue
            p = idx[np.argmax(act[idx])]
            if p - pre >= 0 and p + post <= len(acc):
                out.append((p, acc[p - pre:p + post]))
        return out
    thresh = thresh if thresh is not None else np.percentile(act, 90)
    last = -10 ** 9
    for p in np.argsort(-act):
        if act[p] < thresh:
            break
        if abs(p - last) < min_gap_s * RATE or any(abs(p - q) < min_gap_s * RATE for q, _ in out):
            continue
        if p - pre >= 0 and p + post <= len(acc):
            out.append((p, acc[p - pre:p + post]))
            last = p
    return out


class Recordings:
    """data/gestures/<label>_<time>.npz files: t (ms), acc (n,3), cues (s from start)."""

    @staticmethod
    def save(label, t_ms, acc, cues):
        DATA.mkdir(parents=True, exist_ok=True)
        path = DATA / ("%s_%d.npz" % (label, int(time.time())))
        np.savez(path, t=np.asarray(t_ms, float), acc=np.asarray(acc, float), cues=np.asarray(cues, float))
        return path

    @staticmethod
    def dataset():
        X, y = [], []
        for f in sorted(DATA.glob("*.npz")):
            label = f.stem.rsplit("_", 1)[0]
            d = np.load(f)
            if len(d["t"]) < RATE:
                continue
            if label == "none" or label.startswith("pose_"):
                # everything that is not a gesture: a window every 0.3 s of normal movement
                # and of the held poses (389 examples from the first real session)
                label = "none"
                pre, post = int(PRE_S * RATE), int(POST_S * RATE)
                wins = [(q, d["acc"][q - pre:q + post]) for q in range(pre, len(d["acc"]) - post, int(0.3 * RATE))]
            else:
                wins = peak_windows(d["t"], d["acc"], cue_times=list(d["cues"]))
                # the quiet moment before each cue, same arm position, is NOT the gesture
                tt = (d["t"] - d["t"][0]) / 1000.0
                pre, post = int(PRE_S * RATE), int(POST_S * RATE)
                for c in d["cues"]:
                    q = int(np.searchsorted(tt, c - 0.95))
                    if q - pre >= 0 and q + post <= len(tt):
                        X.append(features(d["acc"][q - pre:q + post]))
                        y.append("none")
            for _, w in wins:
                X.append(features(w))
                y.append(label)
        return np.array(X), np.array(y)

    @staticmethod
    def counts():
        c = {}
        for f in DATA.glob("*.npz"):
            label = f.stem.rsplit("_", 1)[0]
            c[label] = c.get(label, 0) + 1
        return c


def train():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    import pickle
    X, y = Recordings.dataset()
    labels, counts = np.unique(y, return_counts=True) if len(y) else ([], [])
    report = {"samples": {str(l): int(c) for l, c in zip(labels, counts)}}
    if len(labels) < 2 or min(counts) < 3:
        report["error"] = "Need at least 3 examples of each gesture and some normal movement."
        return report
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=1, class_weight="balanced", random_state=0)
    folds = min(5, int(min(counts)))
    pred = cross_val_predict(clf, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=0))
    report["accuracy"] = round(float((pred == y).mean()), 3)
    report["per_label"] = {}
    for l in labels:
        m = y == l
        report["per_label"][str(l)] = {"recall": round(float((pred[m] == l).mean()), 2),
                                       "false_alarms": int(((pred == l) & (y != l)).sum())}
    clf.fit(X, y)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    MODEL.write_bytes(pickle.dumps(clf))
    (DATA / "report.json").write_text(json.dumps(report, indent=2))
    return report


class LiveDetector:
    """Feed samples as they arrive; returns (label, prob) when a gesture is recognised."""

    def __init__(self, min_prob=0.6, refractory_s=1.0):
        import pickle
        self.clf = pickle.loads(MODEL.read_bytes())
        self.min_prob, self.refractory = min_prob, refractory_s
        self.buf = []                     # (t_ms, ax, ay, az)
        self.last_fire = -1e9
        self.last_check = -1e9
        self.thresh = self._threshold()

    def _threshold(self):
        """Half the weakest real gesture's peak activity: anything quieter is ignored."""
        peaks = []
        for f in DATA.glob("*.npz"):
            label = f.stem.rsplit("_", 1)[0]
            if label == "none" or label.startswith("pose_"):
                continue
            d = np.load(f)
            act = activity(d["acc"])            # on the whole recording: no edge effects
            for p, _ in peak_windows(d["t"], d["acc"], cue_times=list(d["cues"])):
                peaks.append(act[p])
        return 0.5 * float(np.percentile(peaks, 10)) if peaks else 0.05

    def feed(self, t_ms, ax, ay, az):
        self.buf.append((t_ms, ax, ay, az))
        if len(self.buf) > RATE * 4:
            del self.buf[: len(self.buf) - RATE * 4]
        now = t_ms / 1000.0                    # band clock: works the same live or replayed
        if now - self.last_check < 0.1 or now - self.last_fire < self.refractory or len(self.buf) < RATE * 2:
            return None
        self.last_check = now
        a = np.array(self.buf)[:, 1:4]
        act = activity(a)
        pre, post = int(PRE_S * RATE), int(POST_S * RATE)
        # the candidate peak must be far enough back that its whole window has arrived
        lo, hi = len(a) - post - int(0.3 * RATE), len(a) - post
        if lo <= pre:
            return None
        p = lo + int(np.argmax(act[lo:hi]))
        if act[p] < self.thresh:
            return None
        w = a[p - pre:p + post]
        probs = self.clf.predict_proba([features(w)])[0]
        i = int(np.argmax(probs))
        label, prob = self.clf.classes_[i], float(probs[i])
        if label != "none" and prob >= self.min_prob:
            self.last_fire = now
            return label, round(prob, 2)
        return None


# ---------------------------------------------------------------- poses (held positions)
#
# Each pose is recorded for ~20 s while the wearer varies it on purpose (angles, sitting,
# standing). Every 0.5 s window of that becomes one example: average gravity direction plus
# how much it wobbled. A random forest learns each pose as a region, not a single angle.

POSE_MODEL = DATA / "pose_model.pkl"
POSE_WIN_S = 0.5


def pose_features(acc):
    m = acc.mean(0)
    u = m / (np.linalg.norm(m) or 1.0)
    return np.concatenate([u, acc.std(0), [np.linalg.norm(m)]])


def pose_dataset():
    X, y = [], []
    win = int(POSE_WIN_S * RATE)
    for f in sorted(DATA.glob("pose_*.npz")):
        label = f.stem.rsplit("_", 1)[0][len("pose_"):]
        acc = np.load(f)["acc"]
        acc = acc[int(1.0 * RATE):]                     # skip the first second (getting into position)
        for i in range(0, len(acc) - win, win // 2):     # 50% overlap
            X.append(pose_features(acc[i:i + win]))
            y.append(label)
    return np.array(X), np.array(y)


def train_poses():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict, GroupKFold, StratifiedKFold
    import pickle
    X, y = pose_dataset()
    labels, counts = np.unique(y, return_counts=True) if len(y) else ([], [])
    report = {"samples": {str(l): int(c) for l, c in zip(labels, counts)}}
    if len(labels) < 2:
        report["error"] = "Record at least two poses."
        return report
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=0)
    # contiguous folds (no shuffling): neighbouring windows overlap, shuffling would be too optimistic
    pred = cross_val_predict(clf, X, y, cv=StratifiedKFold(min(5, int(min(counts))), shuffle=False))
    report["accuracy"] = round(float((pred == y).mean()), 3)
    report["per_label"] = {str(l): {"recall": round(float((pred[y == l] == l).mean()), 2),
                                    "false_alarms": int(((pred == l) & (y != l)).sum())} for l in labels}
    clf.fit(X, y)
    POSE_MODEL.parent.mkdir(parents=True, exist_ok=True)
    POSE_MODEL.write_bytes(pickle.dumps(clf))
    (DATA / "pose_report.json").write_text(json.dumps(report, indent=2))
    return report


class LivePose:
    """Feed samples; .current is the pose that has held for HOLD_S with good confidence."""
    HOLD_S = 0.35

    def __init__(self):
        import pickle
        self.clf = pickle.loads(POSE_MODEL.read_bytes())
        self.buf = []
        self.last_t = -1e9
        self.candidate, self.since = None, 0.0
        self.current, self.prob = "moving", 0.0
        self.raw = None

    def feed(self, t_ms, ax, ay, az):
        self.buf.append((ax, ay, az))
        win = int(POSE_WIN_S * RATE)
        if len(self.buf) > win:
            del self.buf[: len(self.buf) - win]
        now = t_ms / 1000.0
        if now - self.last_t < 0.1 or len(self.buf) < win:
            return self.current
        self.last_t = now
        probs = self.clf.predict_proba([pose_features(np.array(self.buf))])[0]
        i = int(np.argmax(probs))
        label, p = str(self.clf.classes_[i]), float(probs[i])
        self.raw = (label, round(p, 2))
        if p < 0.6 or np.array(self.buf).std(0).max() > 0.25:   # unsure, or arm swinging
            label = "moving"
        if label != self.candidate:
            self.candidate, self.since = label, now
        if now - self.since >= self.HOLD_S:
            self.current, self.prob = label, p
        return self.current
