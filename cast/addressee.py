"""Was this line said to Cast? A small model trained on the wearer's own labelled examples.

The hand-written gate in app.py (loudness, wrist pose, whose voice) logs every line it sees to
data/addressee_log.jsonl. During a labelling session, started from the screen's Details area,
each line also gets a label: 1 in a "to Cast" session (everything the wearer says is a command
or a question), 0 in a "conversation" session (ordinary talk, wrist raised or not).

    .venv/bin/python -m cast.addressee        # train on the labelled lines, compare with the rules

The model sees only what the gate knew when it decided: loudness, pose, voice, timing, and a
few cues from the wording. No audio is logged, and nothing here calls the network.
"""
import json
import pickle
import re
import time

import numpy as np

from .paths import DATA

LOG = DATA / "addressee_log.jsonl"
MODEL = DATA / "addressee_model.pkl"

POSES = ("mouth", "rest_table", "side", "moving")
VOICES = ("wearer", "other", "unknown")
REQUESTS = ("remind me", "can you", "could you", "would you", "will you", "tell me", "what did", "what was",
            "what's", "what is", "who said", "when do", "how long", "cancel", "set a", "set the", "every ",
            "when i ", "when anyone", "when someone", "rename", "that was", "recap", "summarize", "summarise",
            "wake me", "let me know", "note that", "schedule", "in five", "in ten", "minutes", "o'clock")
IMPERATIVES = {"remind", "set", "tell", "cancel", "recap", "rename", "add", "start", "stop", "pause", "schedule",
               "wake", "note", "show", "list", "summarize", "summarise", "delete", "remove", "clear"}

FEATURES = (["rms_db", "rms_missing", "pose_angle", "angle_missing", "margin", "duration", "words", "pending",
             "since_cast"]
            + ["pose_" + p for p in POSES] + ["pose_else"]
            + ["voice_" + v for v in VOICES]
            + ["w_request", "w_question", "w_cast", "w_imperative", "w_you", "w_me", "w_time"])


def wording(text):
    """Cheap cues from the words themselves."""
    t = " ".join(str(text or "").lower().split())
    words = re.findall(r"[a-z']+", t)
    return {
        "w_request": float(any(r in t for r in REQUESTS)),
        "w_question": float("?" in t),
        "w_cast": float("cast" in words),
        "w_imperative": float(bool(words) and words[0] in IMPERATIVES),
        "w_you": float("you" in words),
        "w_me": float("me" in words or "my" in words),
        "w_time": float(bool(re.search(r"\b(\d+|one|two|three|five|ten|twenty|thirty)\s*(second|minute|hour|pm|am)", t))),
    }


def vector(row):
    """One logged line -> the numbers the model sees, in FEATURES order."""
    rms, angle = row.get("rms_db"), row.get("pose_angle")
    f = {
        "rms_db": float(rms) if rms is not None else 30.0, "rms_missing": float(rms is None),
        "pose_angle": float(angle) if angle is not None else 90.0, "angle_missing": float(angle is None),
        "margin": float(row.get("margin") or 0.0), "duration": float(row.get("duration") or 0.0),
        "words": float(row.get("words") or 0), "pending": float(bool(row.get("pending"))),
        "since_cast": float(min(60.0, max(0.0, row.get("since_cast") if row.get("since_cast") is not None else 60.0))),
    }
    pose = row.get("pose")
    for p in POSES:
        f["pose_" + p] = float(pose == p)
    f["pose_else"] = float(pose not in POSES)
    voice = row.get("voice")
    for v in VOICES:
        f["voice_" + v] = float(voice == v)
    f.update(wording(row.get("text")))
    return [f[name] for name in FEATURES]


def log(row, path=None):
    """Append one line of speech (no audio) to the log."""
    path = path or LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def read(path=None):
    rows = []
    try:
        with open(path or LOG) as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return rows


def _scores(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"precision": round(tp / (tp + fp), 2) if tp + fp else None,
            "recall": round(tp / (tp + fn), 2) if tp + fn else None,
            "false_commands": fp, "missed_commands": fn}


def train(log_path=None, model_path=None, quiet=False):
    """Fit on the labelled lines, score it on labelling sessions it has not seen, compare with
    the hand-written rules on the same lines, and save the model. Returns the report."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
    rows = [r for r in read(log_path) if r.get("label") in (0, 1)]
    y = np.array([r["label"] for r in rows])
    report = {"lines": len(rows), "to_cast": int((y == 1).sum()) if len(rows) else 0,
              "conversation": int((y == 0).sum()) if len(rows) else 0}
    if len(rows) < 30 or report["to_cast"] < 10 or report["conversation"] < 10:
        report["error"] = ("Not enough labelled lines yet: need at least 10 said to Cast and 10 of conversation "
                           "(have %d and %d)." % (report["to_cast"], report["conversation"]))
        if not quiet:
            print(report["error"])
        return report
    X = np.array([vector(r) for r in rows])
    groups = np.array([str(r.get("session") or "") for r in rows])
    make = lambda: GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=0.08, subsample=0.9,
                                              random_state=0)
    # a session holds one kind of line only, so hold out sessions in pairs where possible
    if len(set(groups)) >= 4:
        cv, kw, report["tested_on"] = GroupKFold(min(5, len(set(groups)))), {"groups": groups}, "labelling sessions it did not train on"
    else:
        cv, kw, report["tested_on"] = StratifiedKFold(5, shuffle=True, random_state=0), {}, (
            "lines held out at random (record at least two sessions of each kind for a stricter test)")
    try:
        prob = cross_val_predict(make(), X, y, cv=cv, method="predict_proba", **kw)[:, 1]
    except ValueError:                                  # a fold with one class only
        cv, kw = StratifiedKFold(5, shuffle=True, random_state=0), {}
        report["tested_on"] = "lines held out at random"
        prob = cross_val_predict(make(), X, y, cv=cv, method="predict_proba")[:, 1]
    report["model"] = _scores(y, (prob >= 0.5).astype(int))
    report["rules"] = _scores(y, np.array([int(bool((r.get("rule") or {}).get("to_cast"))) for r in rows]))
    clf = make().fit(X, y)
    top = sorted(zip(clf.feature_importances_, FEATURES), reverse=True)[:6]
    report["top_features"] = [(name, round(float(w), 2)) for w, name in top]
    model_path = model_path or MODEL
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(pickle.dumps({"model": clf, "features": list(FEATURES), "trained": time.time(),
                                         "lines": len(rows)}))
    if not quiet:
        print("Labelled lines: %d said to Cast, %d of conversation. Tested on %s." % (
            report["to_cast"], report["conversation"], report["tested_on"]))
        for name in ("model", "rules"):
            s = report[name]
            print("  %-6s precision %s  recall %s  | conversation taken as a command: %d  | commands missed: %d" % (
                name, s["precision"], s["recall"], s["false_commands"], s["missed_commands"]))
        print("  leans most on:", ", ".join("%s %.2f" % t for t in report["top_features"]))
        print("Saved", model_path)
    return report


_loaded = {}


def available(model_path=None):
    return (model_path or MODEL).exists()


def predict(row, model_path=None):
    """Probability that this line was said to Cast, or None if there is no usable model."""
    path = model_path or MODEL
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    if _loaded.get(key, (None, None))[0] != mtime:
        try:
            saved = pickle.loads(path.read_bytes())
            if saved.get("features") != list(FEATURES):
                return None                               # trained with a different set of features
            _loaded[key] = (mtime, saved["model"])
        except Exception:  # noqa: BLE001
            return None
    return float(_loaded[key][1].predict_proba([vector(row)])[0, 1])


if __name__ == "__main__":
    train()
