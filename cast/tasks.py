"""Tasks: something Cast does later, set by voice.

A task is a trigger plus an action.
  trigger  time      at a moment                      {"type": "time", "at": epoch}
           repeat    every N seconds                  {"type": "repeat", "every_s": 1200}
           heard     when someone says something      {"type": "heard", "who": "Hardik"|"anyone",
                                                       "about": "the deadline", "keywords": [...],
                                                       "repeat": False}
           activity  when the wearer starts/stops moving  {"type": "activity", "when": "start_moving"}
  action   say       speak a reminder (repeats until acknowledged with a tap)
           answer    work out an answer from the conversation at that moment and speak it

Saved to data/reminders.json so a restart does not lose them. Entries written by
older versions (a plain timed reminder) still load.
"""
import json
import threading
import time
import uuid
from .paths import DATA

STORE = DATA / "reminders.json"
MAX_RUNS = 50            # a repeating task stops after this many runs
MIN_EVERY_S = 10


def trigger_label(trigger):
    kind = trigger.get("type")
    if kind == "time":
        return "at " + time.strftime("%-I:%M %p", time.localtime(trigger.get("at") or time.time()))
    if kind == "repeat":
        s = int(trigger.get("every_s") or 0)
        if s < 120:
            return "every %d sec" % s
        if s < 7200:
            return "every %d min" % round(s / 60)
        return "every %g hr" % round(s / 3600, 1)
    if kind == "heard":
        who = (trigger.get("who") or "anyone").strip()
        who = "someone" if who.lower() in ("anyone", "someone", "anybody", "") else who
        about = (trigger.get("about") or "").strip() or ", ".join(trigger.get("keywords") or []) or "that"
        return "%s %s mentions %s" % ("whenever" if trigger.get("repeat") else "when", who, about)
    if kind == "activity":
        return {"start_moving": "when you get up", "stop_moving": "when you stop moving"}.get(
            trigger.get("when"), "when you move")
    return ""


def action_label(action, task):
    if action.get("type") == "answer":
        return "tell you: %s" % (action.get("prompt") or task)
    return "remind you: %s" % (action.get("text") or task)


def _upgrade(r):
    """Fill in the fields newer code expects on an entry saved by an older version."""
    if "trigger" not in r:
        r["trigger"] = {"type": "time", "at": r.get("remind_at")}
    if "action" not in r:
        r["action"] = {"type": "say", "text": r.get("task") or "Reminder"}
    r.setdefault("remind_at", r["trigger"].get("at"))
    r.setdefault("runs", 0)
    r["kind_of"] = r["trigger"].get("type", "time")
    r["trigger_label"] = trigger_label(r["trigger"])
    r["action_label"] = action_label(r["action"], r.get("task") or "Reminder")
    return r


class Scheduler:
    def __init__(self, on_due, start=True):
        self.on_due = on_due
        self.lock = threading.Lock()
        self.items = self._load()
        if start:
            threading.Thread(target=self._loop, daemon=True).start()

    def _load(self):
        try:
            items = json.loads(STORE.read_text())
            return [_upgrade(r) for r in items if isinstance(r, dict) and r.get("status") == "scheduled"]
        except (OSError, ValueError):
            return []

    def _save(self):
        STORE.parent.mkdir(exist_ok=True)
        tmp = STORE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.items, indent=2))
        tmp.replace(STORE)

    # ---- adding

    def add(self, task, remind_at, say=None):
        """A plain timed, spoken reminder (the original kind)."""
        return self.add_task(task, {"type": "time", "at": float(remind_at)}, {"type": "say", "text": task}, say)

    def add_task(self, task, trigger, action, say=None, now=None):
        now = now or time.time()
        trigger, action = dict(trigger), dict(action)
        kind = trigger.get("type", "time")
        remind_at = None
        if kind == "time":
            remind_at = float(trigger["at"])
        elif kind == "repeat":
            trigger["every_s"] = max(MIN_EVERY_S, int(trigger.get("every_s") or 0))
            remind_at = now + trigger["every_s"]
        task = task or "Reminder"
        if action.get("type") != "answer":
            action = {"type": "say", "text": action.get("text") or task}
        r = _upgrade({"id": uuid.uuid4().hex[:8], "task": task, "status": "scheduled", "created": now,
                      "trigger": trigger, "action": action, "remind_at": remind_at, "say": say, "runs": 0})
        with self.lock:
            self.items.append(r)
            self._save()
        return dict(r)

    # ---- cancelling

    def cancel_latest(self):
        gone = self.cancel("")
        return gone[0] if gone else None

    def cancel(self, which, pending_tasks_chosen=None):
        """which: "all", or the tasks to cancel (exact task strings), or empty for the latest."""
        with self.lock:
            pending = [r for r in self.items if r["status"] == "scheduled"]
            if which == "all":
                gone = pending
            elif pending_tasks_chosen:
                want = {t.strip().lower() for t in pending_tasks_chosen}
                gone = [r for r in pending if r["task"].strip().lower() in want]
            else:
                gone = [max(pending, key=lambda x: x["created"])] if pending else []
            for r in gone:
                r["status"] = "cancelled"
                self.items.remove(r)
            if gone:
                self._save()
            return [dict(r) for r in gone]

    # ---- reading

    def pending(self):
        """Scheduled tasks: the timed ones first, soonest first, then the ones waiting on an event."""
        with self.lock:
            items = [dict(r) for r in self.items if r["status"] == "scheduled"]
        return sorted(items, key=lambda r: (r["remind_at"] is None, r["remind_at"] or 0, r["created"]))

    def waiting(self, kind):
        """Scheduled tasks whose trigger is an event of this kind ("heard" or "activity")."""
        with self.lock:
            return [dict(r) for r in self.items if r["status"] == "scheduled" and r["trigger"].get("type") == kind]

    # ---- firing

    def fire(self, task_id, **extra):
        """An event trigger happened. One-shot tasks are removed; "whenever" tasks stay."""
        with self.lock:
            r = next((x for x in self.items if x["id"] == task_id and x["status"] == "scheduled"), None)
            if r is None:
                return None
            r["runs"] = r.get("runs", 0) + 1
            fired = dict(r, status="due", **extra)
            if not r["trigger"].get("repeat") or r["runs"] >= MAX_RUNS:
                self.items.remove(r)
            self._save()
        self._call(fired)
        return fired

    def fire_activity(self, when):
        return [self.fire(r["id"]) for r in self.waiting("activity") if r["trigger"].get("when") == when]

    def tick(self, now=None):
        """Fire everything whose time has come. Called twice a second by the loop."""
        now = now or time.time()
        due = []
        with self.lock:
            for r in list(self.items):
                if r["status"] != "scheduled" or r["remind_at"] is None or r["remind_at"] > now:
                    continue
                r["runs"] = r.get("runs", 0) + 1
                due.append(dict(r, status="due"))
                if r["trigger"].get("type") == "repeat" and r["runs"] < MAX_RUNS:
                    every = r["trigger"]["every_s"]
                    while r["remind_at"] <= now:          # skip runs missed while the app was off
                        r["remind_at"] += every
                else:
                    self.items.remove(r)
            if due:
                self._save()
        for r in due:
            self._call(r)
        return due

    def _call(self, r):
        try:
            self.on_due(r)
        except Exception as e:  # noqa: BLE001
            print("[scheduler] on_due failed:", e)

    def _loop(self):
        while True:
            self.tick()
            time.sleep(0.5)


def spoken_time(epoch):
    if epoch is None:
        return "when it happens"
    t = time.localtime(epoch)
    delta = epoch - time.time()
    if delta < 90:
        return "in %d seconds" % max(1, int(delta))
    if delta < 3600:
        return "in %d minutes" % round(delta / 60)
    return "at " + time.strftime("%-I:%M", t)


def keyword_hit(text, keywords):
    """Cheap first filter for a "heard" trigger: does the line contain any keyword?"""
    low = " " + "".join(c if c.isalnum() else " " for c in text.lower()) + " "
    for k in keywords or []:
        k = "".join(c if c.isalnum() else " " for c in str(k).lower()).strip()
        if len(k) >= 3 and (" " + k in low):              # word-start match: "deadline" hits "deadlines"
            return True
    return False
