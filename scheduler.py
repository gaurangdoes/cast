"""Reminders: scheduled by voice, fired by Cast out loud and on the screen.

Saved to data/reminders.json so a restart of app.py does not lose them.
"""
import json
import threading
import time
import uuid
from pathlib import Path

STORE = Path(__file__).resolve().parent / "data" / "reminders.json"


class Scheduler:
    def __init__(self, on_due):
        self.on_due = on_due
        self.lock = threading.Lock()
        self.items = self._load()
        threading.Thread(target=self._loop, daemon=True).start()

    def _load(self):
        try:
            items = json.loads(STORE.read_text())
            return [r for r in items if r.get("status") == "scheduled"]
        except (OSError, ValueError):
            return []

    def _save(self):
        STORE.parent.mkdir(exist_ok=True)
        STORE.write_text(json.dumps(self.items, indent=2))

    def add(self, task, remind_at, say=None):
        r = {"id": uuid.uuid4().hex[:8], "task": task or "Reminder", "remind_at": float(remind_at),
             "status": "scheduled", "say": say, "created": time.time()}
        with self.lock:
            self.items.append(r)
            self._save()
        return r

    def cancel_latest(self):
        with self.lock:
            pending = [r for r in self.items if r["status"] == "scheduled"]
            if not pending:
                return None
            r = max(pending, key=lambda x: x["created"])
            r["status"] = "cancelled"
            self.items.remove(r)
            self._save()
            return r

    def pending(self):
        with self.lock:
            return sorted((dict(r) for r in self.items if r["status"] == "scheduled"), key=lambda r: r["remind_at"])

    def _loop(self):
        while True:
            now = time.time()
            due = []
            with self.lock:
                for r in self.items:
                    if r["status"] == "scheduled" and r["remind_at"] <= now:
                        r["status"] = "due"
                        due.append(dict(r))
                if due:
                    self.items = [r for r in self.items if r["status"] == "scheduled"]
                    self._save()
            for r in due:
                try:
                    self.on_due(r)
                except Exception as e:  # noqa: BLE001
                    print("[scheduler] on_due failed:", e)
            time.sleep(0.5)


def spoken_time(epoch):
    t = time.localtime(epoch)
    delta = epoch - time.time()
    if delta < 90:
        return "in %d seconds" % max(1, int(delta))
    if delta < 3600:
        return "in %d minutes" % round(delta / 60)
    return "at " + time.strftime("%-I:%M", t)
