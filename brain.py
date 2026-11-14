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
    out = json.loads(data["candidates"][0]["content"]["parts"][0]["text"])
    if isinstance(out, list):                 # Gemini sometimes wraps the object in a list
        out = out[0] if out else {}
    return out if isinstance(out, dict) else {}


def ask_json(prompt, fast=False):
    order = ((FALLBACK, 5), (PRIMARY, 8)) if fast else ((PRIMARY, 8), (FALLBACK, 6))
    for model, timeout in order:
        try:
            return _gemini(prompt, model, timeout)
        except Exception as e:  # noqa: BLE001 - any failure falls through to the next model
            print("[brain] %s failed: %s" % (model, str(e)[:100]))
    return None


def _now():
    return dt.datetime.now().astimezone()


COMMAND_PROMPT = """You are Cast, a voice assistant in a wristband. The wearer raised the band to
their mouth and gave a command. Current local time: {now}.

Current tasks and reminders (oldest first): {reminders}

Command: "{text}"

Return JSON with exactly these keys:
- "intent": "task" if they want something to happen later (a reminder, an alert, a recap, anything
  with "remind me", "tell me when", "every", "when X happens"), "cancel_reminder" if they want to
  cancel one, "question" if they are asking NOW about something that was said or about their
  tasks (e.g. "what did she just say?", "when do I need to move the car?"), "label_speaker" if
  they are telling you who someone is (e.g. "that was Hardik", "the person I'm talking to is Maya"),
  otherwise "other".
- "name": for "label_speaker", the person's first name; otherwise empty.
- "cancel": for "cancel_reminder" only: the list of tasks to cancel, copied exactly from the
  current tasks above. "both", "all", "every" or "my reminders" means all of them. If they
  name one ("the parking one"), just that one. If unclear, the most recently created (listed last).
For "task" (leave these empty otherwise):
- "task": short label, e.g. "Move the car", "Recap decisions", "Hardik mentions the deadline".
- "trigger_type": one of
    "time"     once, at a moment ("at 4", "in 20 minutes", "5 minutes before it ends")
    "repeat"   again and again ("every 20 minutes")
    "heard"    when someone says something ("when Hardik mentions the deadline", "if anyone
               brings up pricing", "whenever someone says my name")
    "activity" when the wearer moves ("when I get up", "when I start walking", "when I sit down")
- "remind_at": for "time": ISO 8601 local time with offset. Work it out from what they said, e.g.
  "I parked, it ends in 2 hours, remind me 5 minutes before" means now + 1h55m.
- "every_seconds": for "repeat": the interval in seconds (a number).
- "who": for "heard": the first name of the person who must say it, or "anyone".
- "about": for "heard": what must be mentioned, a short noun phrase ("the deadline").
- "keywords": for "heard": 3 to 8 lowercase words or short phrases likely to appear if it is
  mentioned, including synonyms ("deadline", "due", "submit", "cutoff").
- "heard_repeat": for "heard": true only if they said whenever / every time / each time.
- "activity_when": for "activity": "start_moving" (get up, stand up, start walking, leave) or
  "stop_moving" (sit down, stop, arrive).
- "action_type": "say" to speak a fixed reminder, or "answer" when what to say must be worked out
  at that moment from the conversation (a recap, a summary, "what I've promised people", "how
  long is left", "what did we decide").
- "action_text": for "say": the reminder as Cast should speak it, e.g. "Move the car".
- "action_prompt": for "answer": the question Cast should answer at that moment, written as a
  full question, e.g. "What have we decided in this conversation?".
- "say": one short spoken reply (under 15 words) confirming what you will do, with any time
  spoken naturally ("at 2:25", "in 30 seconds", "every 20 minutes").
Speech may include false starts, filler and self-corrections ("actually...", "no wait"). Use the
final intent. If the command contains any request for later, the intent is "task" even if they
also ask for something you can't do; set the task and mention briefly what you can't do.
"""

ANSWER_PROMPT = """You are Cast, a voice assistant in a wristband. The wearer raised the band to
their mouth and asked a question (or scheduled one for now). Answer only from what Cast heard and
the tasks below.
Current local time: {now}.

What Cast heard recently (oldest first; "You" is the wearer, others are labelled by speaker):
{transcript}

Scheduled tasks and reminders:
{reminders}

Question: "{text}"

Return JSON {{"say": "<spoken answer, under 30 words, say who said it when relevant>"}}.
If the answer is not in what was heard, say so briefly.
"""


def interpret_command(text, reminders=()):
    """Returns the model's dict. For intent "task" it also carries "trigger" and "action" ready
    for the scheduler, and "remind_at_epoch" for timed tasks."""
    now = _now()
    out = ask_json(COMMAND_PROMPT.format(now=now.isoformat(timespec="seconds"), text=text,
                                         reminders=json.dumps(list(reminders)) if reminders else "none"), fast=True)
    if not out:
        out = regex_reminder(text, now) or {"intent": "other", "say": "Sorry, I couldn't do that."}
    if out.get("intent") == "reminder":            # the older name for the same thing
        out["intent"] = "task"
    if out.get("intent") == "task":
        out = build_task(out, text, now)
    return out


def build_task(out, text, now):
    kind = str(out.get("trigger_type") or "time").strip().lower()
    task = (out.get("task") or "").strip() or "Reminder"
    if str(out.get("action_type") or "").strip().lower() == "answer" and (out.get("action_prompt") or "").strip():
        action = {"type": "answer", "prompt": out["action_prompt"].strip()}
    else:
        action = {"type": "say", "text": (out.get("action_text") or "").strip() or task}
    trigger = None
    if kind == "repeat":
        try:
            every = int(float(out.get("every_seconds") or 0))
        except (TypeError, ValueError):
            every = 0
        if every > 0:
            trigger = {"type": "repeat", "every_s": every}
    elif kind == "heard":
        kws = out.get("keywords") if isinstance(out.get("keywords"), list) else []
        common = {"you", "your", "name", "time", "date", "that", "this", "with", "about", "have", "will", "they"}
        kws = [k for k in (str(k).strip().lower() for k in kws) if len(k) >= 4 and k not in common][:10]
        about = (out.get("about") or "").strip()
        if not kws and about:
            kws = [w for w in re.findall(r"[a-z']+", about.lower()) if len(w) > 3]
        if kws:
            who = (out.get("who") or "anyone").strip() or "anyone"
            trigger = {"type": "heard", "who": who, "about": about or kws[0], "keywords": kws,
                       "repeat": out.get("heard_repeat") is True}
    elif kind == "activity":
        when = str(out.get("activity_when") or "").strip().lower()
        if when in ("start_moving", "stop_moving"):
            trigger = {"type": "activity", "when": when}
    if trigger is None:                            # a time, or anything that failed to parse
        try:
            when = dt.datetime.fromisoformat(out["remind_at"])
            if when.tzinfo is None:
                when = when.astimezone()
            trigger = {"type": "time", "at": when.timestamp()}
        except (KeyError, ValueError, TypeError):
            fallback = regex_reminder(text, now)
            if not fallback:
                return {"intent": "other", "say": "I didn't catch when to do that."}
            trigger = {"type": "time", "at": fallback["remind_at_epoch"]}
            out.setdefault("say", fallback["say"])
    out.update(intent="task", task=task, trigger=trigger, action=action)
    if trigger["type"] == "time":
        out["remind_at_epoch"] = trigger["at"]
    return out


HEARD_PROMPT = """A wristband assistant is waiting for this to happen: {who} mentions {about}.

A line was just said in the conversation. Speaker: {speaker}. Line: "{line}"

Does this line count? It counts only if the speaker is actually talking about {about}
(not merely using a similar word for something else){who_rule}.
Return JSON {{"match": true or false}}."""


def heard_match(line, speaker, who, about):
    """The second check for a "heard" trigger, after a keyword hit. None if Gemini can't be reached."""
    anyone = (who or "anyone").strip().lower() in ("anyone", "someone", "anybody", "")
    out = ask_json(HEARD_PROMPT.format(
        who="someone" if anyone else who, about=about, speaker=speaker or "unknown", line=line.replace('"', "'"),
        who_rule="" if anyone else ", and the speaker is %s" % who), fast=True)
    if not isinstance(out, dict) or "match" not in out:
        return None
    return out.get("match") is True


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
    return {"intent": "task", "task": "Reminder", "remind_at": when.isoformat(),
            "remind_at_epoch": when.timestamp(), "say": "Okay, I'll remind you at %s." % when.strftime("%-I:%M")}


NAME_PROMPT = """One line from a live conversation transcript: "{text}"

Return JSON {{"addressed": "<first name of the person the speaker is talking TO, if they say it,
e.g. 'hey Hardik, how's it going' -> Hardik>", "self_name": "<the speaker's own first name if they
state it, e.g. 'I'm Maya' or 'this is Maya' -> Maya>"}}. Use "" when absent. Names of people who are
only talked ABOUT do not count. Never guess."""


def find_names(text):
    out = ask_json(NAME_PROMPT.format(text=text.replace('"', "'")), fast=True) or {}
    clean = lambda v: v.strip().split()[0].strip(".,!?").title() if isinstance(v, str) and v.strip() else ""
    return clean(out.get("addressed")), clean(out.get("self_name"))
