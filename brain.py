"""Gemini: turns spoken commands into actions and answers questions from what was heard.

Model: Gemini 4 (gemini-4-hydrogen) first, falling back to a fast Flash Lite model if it is
slow or fails. Measured on the venue wifi: hydrogen ~3 s, 3.5-flash-lite ~1 s.
If Gemini is unreachable, a small regex parser still handles "remind me in N minutes".
"""
import datetime as dt
import json
import os
import re
import urllib.request

PRIMARY = os.environ.get("CAST_MODEL", "gemini-4-hydrogen")
FALLBACK = os.environ.get("CAST_FALLBACK_MODEL", "gemini-3.5-flash-lite")
URL = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent"


def _gemini(prompt, model, timeout):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("no GEMINI_API_KEY")
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json"}}
    req = urllib.request.Request(URL % model, data=json.dumps(body).encode(),
                                 headers={"x-goog-api-key": key, "Content-Type": "application/json"})
    data = json.load(urllib.request.urlopen(req, timeout=timeout))
    return json.loads(data["candidates"][0]["content"]["parts"][0]["text"])


def ask_json(prompt):
    for model, timeout in ((PRIMARY, 8), (FALLBACK, 6)):
        try:
            return _gemini(prompt, model, timeout)
        except Exception as e:  # noqa: BLE001 - any failure falls through to the next model
            print("[brain] %s failed: %s" % (model, str(e)[:100]))
    return None


def _now():
    return dt.datetime.now().astimezone()


COMMAND_PROMPT = """You are Cast, a voice assistant in a wristband. The wearer raised the band to
their mouth and gave a command. Current local time: {now}.

Command: "{text}"

Return JSON with exactly these keys:
- "intent": "reminder" if they want to be reminded or alerted later, "cancel_reminder" if they
  want to cancel one, "question" if they are asking about something that was said or about their
  reminders (e.g. "what did she just say?", "when do I need to move the car?"), otherwise "other".
- "task": short label for what to remind them about (empty if not a reminder), e.g. "Move the car".
- "remind_at": ISO 8601 local time with offset for when to remind them (empty if not a reminder).
  Work out times from what they said, e.g. "I parked, it ends in 2 hours, remind me 5 minutes
  before" means now + 1h55m. "in 30 seconds" means now + 30s.
- "say": one short spoken reply (under 15 words) confirming what you will do, with the time
  spoken naturally ("at 2:25", "in 30 seconds").
Speech may include false starts, filler and self-corrections ("actually...", "no wait"). Use the
final intent. If the command contains any reminder request, the intent is "reminder" even if they
also ask for something you can't do; set the reminder and mention briefly what you can't do.
"""

ANSWER_PROMPT = """You are Cast, a voice assistant in a wristband. The wearer raised the band to
their ear and asked a question. Answer only from what Cast heard and the reminders below.
Current local time: {now}.

What Cast heard recently (oldest first; "You" is the wearer, others are labelled by speaker):
{transcript}

Scheduled reminders:
{reminders}

Question: "{text}"

Return JSON {{"say": "<spoken answer, under 30 words, say who said it when relevant>"}}.
If the answer is not in what was heard, say so briefly.
"""


def interpret_command(text):
    now = _now()
    out = ask_json(COMMAND_PROMPT.format(now=now.isoformat(timespec="seconds"), text=text))
    if not out:
        out = regex_reminder(text, now) or {"intent": "other", "say": "Sorry, I couldn't do that."}
    if out.get("intent") == "reminder":
        try:
            when = dt.datetime.fromisoformat(out["remind_at"])
            if when.tzinfo is None:
                when = when.astimezone()
            out["remind_at_epoch"] = when.timestamp()
        except (KeyError, ValueError, TypeError):
            fallback = regex_reminder(text, now)
            if not fallback:
                return {"intent": "other", "say": "I didn't catch when to remind you."}
            out = fallback
    return out


def answer_question(text, transcript_lines, reminder_lines):
    out = ask_json(ANSWER_PROMPT.format(
        now=_now().isoformat(timespec="seconds"), text=text,
        transcript="\n".join(transcript_lines[-60:]) or "(nothing yet)",
        reminders="\n".join(reminder_lines) or "(none)"))
    return (out or {}).get("say") or "Sorry, I couldn't answer that."


_UNITS = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600}


def regex_reminder(text, now):
    """Offline fallback: 'in N minutes/hours', optionally 'N minutes before'."""
    t = text.lower()
    m = re.search(r"in (\d+|an?|one|two|three|five|ten|thirty) ?(second|sec|minute|min|hour|hr)s?", t)
    if not m:
        return None
    words = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "five": 5, "ten": 10, "thirty": 30}
    n = int(m.group(1)) if m.group(1).isdigit() else words[m.group(1)]
    seconds = n * _UNITS[m.group(2)]
    b = re.search(r"(\d+) ?(second|sec|minute|min|hour|hr)s? before", t)
    if b:
        seconds -= int(b.group(1)) * _UNITS[b.group(2)]
    when = now + dt.timedelta(seconds=max(5, seconds))
    return {"intent": "reminder", "task": "Reminder", "remind_at": when.isoformat(),
            "remind_at_epoch": when.timestamp(), "say": "Okay, I'll remind you at %s." % when.strftime("%-I:%M")}
