import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { useCast, type Line } from './useCast'
import type { ActionEvent, Banner, Card, ConnState, Mode, Pose, Reminder, TaskKind } from './types'

const POSE_NAME: Record<string, string> = {
  mouth: 'Wrist to mouth', ear: 'Wrist raised', rest_table: 'Resting', side: 'At your side',
  chest: 'On chest', extended: 'Arm out', glance: 'Glance', moving: 'Moving',
}
const SPEAKER_COLORS = 5
const TO_CAST: ReadonlySet<string> = new Set(['command', 'ask', 'note'])
const isToCast = (mode: string | undefined) => !!mode && TO_CAST.has(mode)

/* ---------- small helpers ---------- */

function useStoredFlag(key: string, fallback = false): [boolean, () => void] {
  const [on, setOn] = useState(() => {
    try { return (localStorage.getItem(key) ?? (fallback ? '1' : '0')) === '1' } catch { return fallback }
  })
  useEffect(() => {
    try { localStorage.setItem(key, on ? '1' : '0') } catch { /* storage unavailable */ }
  }, [key, on])
  const toggle = useCallback(() => setOn((v) => !v), [])
  return [on, toggle]
}

interface Who { label: string; kind: 'you' | 'person' | 'unknown'; color: number; initial: string }

function whoOf(speaker: number | null | undefined, isWearer: boolean | undefined, names: Record<string, string>, fallback?: string | null, guess?: string | null): Who {
  if (isWearer) return { label: 'You', kind: 'you', color: 0, initial: 'Y' }
  if (speaker == null || speaker < 0) return { label: guess ? `Likely ${guess}` : 'Someone', kind: 'unknown', color: 0, initial: guess ? guess.charAt(0).toUpperCase() : '?' }
  const label = names[String(speaker)] || fallback || `Person ${speaker}`
  const num = /^(?:Person|Speaker)\s+(\d+)$/i.exec(label)
  return { label, kind: 'person', color: Math.abs(speaker) % SPEAKER_COLORS, initial: num ? num[1] : label.trim().charAt(0).toUpperCase() || '?' }
}

const PATHS: Record<string, ReactNode> = {
  mic: <><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21" /></>,
  micOff: <><path d="M15 10.5V6a3 3 0 0 0-5.7-1.3M9 9v2a3 3 0 0 0 4.6 2.5M5.5 11a6.5 6.5 0 0 0 10.3 5.3M18.5 11c0 .9-.2 1.8-.5 2.6M12 17.5V21M4 3l16 18" /></>,
  ear: <><path d="M7 9.5a5 5 0 0 1 10 0c0 2.6-2.2 3.4-2.9 5.2-.6 1.6-.6 3.8-3 3.8-1.5 0-2.4-1-2.6-2.3" /><path d="M10 10a2 2 0 0 1 4 0c0 1-1 1.5-1.5 2.2" /></>,
  clock: <><circle cx="12" cy="12" r="8.5" /><path d="M12 7.5V12l3 2" /></>,
  repeat: <path d="M4.5 11a7 7 0 0 1 12.3-3.5L19 9.5M19 5v4.5h-4.5M19.5 13a7 7 0 0 1-12.3 3.5L5 14.5M5 19v-4.5h4.5" />,
  activity: <><circle cx="13" cy="4.5" r="1.8" /><path d="M9 21l2.5-6-3-2.5 1.5-5 4 1.5 2 3h3M6.5 11l3-3.5M14.5 21l-.5-4.5" /></>,
  check: <path d="M5 12.5l4.5 4.5L19 7.5" />,
  x: <path d="M6.5 6.5l11 11M17.5 6.5l-11 11" />,
  person: <><circle cx="12" cy="8" r="3.5" /><path d="M5 20a7 7 0 0 1 14 0" /></>,
  quote: <path d="M5 13h4v4H5zM14 13h4v4h-4zM5 13a5 5 0 0 1 4-5M14 13a5 5 0 0 1 4-5" />,
  raise: <path d="M12 19V6M6.5 11.5L12 6l5.5 5.5" />,
  pause: <path d="M9 6v12M15 6v12" />,
  tap: <><circle cx="12" cy="12" r="3" /><circle cx="12" cy="12" r="8" strokeDasharray="2.5 3.5" /></>,
  sliders: <><path d="M4 8h9M17 8h3M4 16h3M11 16h9" /><circle cx="15" cy="8" r="2" /><circle cx="9" cy="16" r="2" /></>,
}

function Icon({ name }: { name: string }) {
  return (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {PATHS[name]}
    </svg>
  )
}

/* ---------- app ---------- */

export default function App() {
  const { state, dismissBanner, rename, assign, label } = useCast()
  const [details, toggleDetails] = useStoredFlag('cast.details')
  const raised = state.pose === 'mouth' || state.pose === 'ear' || state.mode === 'command' || state.mode === 'ask'
  return (
    <>
      {state.banner && <DueBanner key={state.banner.key} banner={state.banner} onDismiss={dismissBanner} />}
      {state.gesture && <GestureToast key={state.gesture.at} label={state.gesture.label} />}
      <div className={`app${state.micOn ? '' : ' mic-off'}`}>
        <header className="topbar">
          <div className="brand">Dextra</div>
          <div className="tagline">Your wristband assistant</div>
          <div className="topbar-right">
            <ConnPill conn={state.conn} />
            <div className={`pill ${state.micOn ? 'pill-on' : 'pill-off'}`}>
              <Icon name={state.micOn ? 'mic' : 'micOff'} />
              {state.micOn ? 'Listening' : 'Off the record'}
            </div>
            <button type="button" className="toggle" aria-pressed={details} onClick={toggleDetails} title="Show per-line details for builders">
              <Icon name="sliders" />
              Details
            </button>
          </div>
        </header>
        {!state.micOn && (
          <div className="offrecord" role="status">
            <Icon name="micOff" />
            <b>Off the record</b>
            <span>The microphone is off. Double-tap or close your fist to resume.</span>
          </div>
        )}
        {state.labelling && <LabellingBar session={state.labelling} onStop={() => label('off')} />}
        <main className="cols">
          <StatusPanel
            labelling={state.labelling} onLabel={label}
            raised={raised} micOn={state.micOn} pose={state.pose} mode={state.mode} angle={state.angle}
            poseSeen={state.poseSeen} levelPct={state.levelPct} wearer={state.wearer} ignored={state.ignored} details={details}
          />
          <TranscriptPanel lines={state.lines} partial={state.partial} names={state.names} details={details} micOn={state.micOn} onRename={rename} onAssign={assign} />
          <section className="side">
            <TasksPanel reminders={state.reminders} />
            <ActivityPanel cards={state.cards} names={state.names} />
          </section>
        </main>
      </div>
    </>
  )
}

const GESTURES: Record<string, string> = {
  fist: 'Fist', thumbs_up: 'Thumbs up', double_tap: 'Double-tap', tap: 'Tap',
}

function GestureToast({ label }: { label: string }) {
  const [show, setShow] = useState(true)
  useEffect(() => { const t = window.setTimeout(() => setShow(false), 1800); return () => window.clearTimeout(t) }, [])
  if (!show) return null
  return (
    <div className="gtoast" role="status">
      <span className="gtoast-k">Gesture</span>
      {GESTURES[label] ?? String(label).replace(/_/g, ' ')}
    </div>
  )
}

function ConnPill({ conn }: { conn: ConnState }) {
  const label = conn === 'live' ? 'Connected' : conn === 'connecting' ? 'Connecting…' : 'Reconnecting…'
  return <div className={`pill conn ${conn}`}><span className="dot" />{label}</div>
}

/* ---------- left: status ---------- */

function LabellingBar({ session, onStop }: { session: { mode: 'cast' | 'chat'; until: number }; onStop: () => void }) {
  const [now, setNow] = useState(() => Date.now() / 1000)
  useEffect(() => { const t = window.setInterval(() => setNow(Date.now() / 1000), 500); return () => window.clearInterval(t) }, [])
  const left = Math.max(0, Math.ceil(session.until - now))
  return (
    <div className="labelling" role="status">
      <b>{session.mode === 'cast' ? 'Recording examples: talk TO Dextra' : 'Recording examples: ordinary conversation'}</b>
      <span>{session.mode === 'cast'
        ? 'Give commands and ask questions the way you normally would: at your mouth, softer, slower. Dextra will not act.'
        : 'Just talk, with your wrist up and down, gesturing. Nothing here is meant for Dextra.'}</span>
      <span className="labelling-left">{left}s</span>
      <button type="button" className="btn" onClick={onStop}>Stop</button>
    </div>
  )
}

function StatusPanel(p: {
  raised: boolean; micOn: boolean; pose: Pose | null; mode: Mode; angle: number | null
  poseSeen: boolean; levelPct: number; wearer: number | null; ignored: number; details: boolean
  labelling: { mode: 'cast' | 'chat'; until: number } | null; onLabel: (mode: 'cast' | 'chat' | 'off') => void
}) {
  const state = !p.micOn ? 'off' : p.raised ? 'raised' : 'ambient'
  const title = state === 'off' ? 'Microphone off' : state === 'raised' ? 'Wrist raised' : 'Listening in the background'
  const sub = state === 'off' ? 'Nothing is being heard' : state === 'raised' ? 'Talking to Dextra' : 'Remembering, not acting'
  const level = p.micOn ? p.levelPct : 0
  return (
    <section className="card-panel status-panel">
      <div className={`wrist ${state}`}>
        <div className="wrist-icon"><Icon name={state === 'off' ? 'micOff' : state === 'raised' ? 'mic' : 'ear'} /></div>
        <div className="wrist-title">{title}</div>
        <div className="wrist-sub">{sub}</div>
        <div className="level" role="meter" aria-label="Microphone level" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(level)}>
          <div style={{ width: `${level}%` }} />
        </div>
      </div>

      {p.details && (
        <dl className="debug">
          <dt>pose</dt><dd>{p.pose ? (POSE_NAME[p.pose] ?? p.pose) : '–'}</dd>
          <dt>mode</dt><dd>{p.mode}</dd>
          <dt>angle</dt><dd>{p.poseSeen && p.angle != null ? `${p.angle}°` : '–'}</dd>
          <dt>wearer</dt><dd>{p.wearer != null ? Number(p.wearer).toFixed(2) : '–'}</dd>
        </dl>
      )}
      {p.details && (
        <div className="teach">
          <h2>Teach the command gate</h2>
          <p>Record examples, then run <code>python -m cast.addressee</code>. Dextra does not act while recording.</p>
          <div className="teach-buttons">
            <button type="button" className="btn" disabled={!!p.labelling} onClick={() => p.onLabel('cast')}>To Dextra · 90 s</button>
            <button type="button" className="btn" disabled={!!p.labelling} onClick={() => p.onLabel('chat')}>Conversation · 120 s</button>
          </div>
        </div>
      )}

      <div className="howto">
        <h2>How to use it</h2>
        <ul>
          <li><span className="howto-icon"><Icon name="raise" /></span><div><b>Raise your wrist</b><span>to talk to Dextra</span></div></li>
          <li><span className="howto-icon"><Icon name="pause" /></span><div><b>Double-tap or fist</b><span>to pause the microphone</span></div></li>
          <li><span className="howto-icon"><Icon name="tap" /></span><div><b>Tap</b><span>to acknowledge</span></div></li>
        </ul>
      </div>

      {p.ignored > 0 && (
        <div className="ignored">{p.ignored} background {p.ignored === 1 ? 'line' : 'lines'} filtered</div>
      )}
    </section>
  )
}

/* ---------- centre: transcript ---------- */

function SpeakerName({ label, speaker, segId, editable, known, onRename, onAssign }: {
  label: string; speaker: number; segId?: string; editable: boolean; known: string[]
  onRename: (speaker: number, name: string) => void; onAssign: (seg: string, name: string) => void
}) {
  const [editing, setEditing] = useState(false)
  const unknown = speaker == null || speaker < 0
  const [value, setValue] = useState(unknown ? '' : label)
  const closed = useRef(false)
  if (!editable || (unknown && !segId)) return <span className="who">{label}</span>
  if (!editing) {
    return (
      <button type="button" className={`who who-edit${unknown ? ' who-ask' : ''}`}
        title={unknown ? 'Click to say who this was' : 'Click to correct who said this, or rename them'}
        onClick={() => { closed.current = false; setValue(unknown ? '' : label); setEditing(true) }}>
        {label}{unknown && <span className="who-hint"> · who was this?</span>}
      </button>
    )
  }
  const save = () => {
    if (closed.current) return                 // Enter then blur, or Escape then blur: act once
    closed.current = true
    const v = value.trim()
    if (v && unknown && segId) onAssign(segId, v)
    else if (v && v !== label) onRename(speaker, v)
    setEditing(false)
  }
  if (unknown && segId) {
    const pick = (name: string) => { closed.current = true; onAssign(segId, name); setEditing(false) }
    return (
      <span className="who-pick" role="group" aria-label="Who said this">
        {['You', ...known].map((k) => (
          <button key={k} type="button" className="who-chip" onMouseDown={(e) => e.preventDefault()}
            onClick={() => pick(k === 'You' ? 'me' : k)}>{k}</button>
        ))}
        <input className="who-input who-new" autoFocus value={value} maxLength={30} placeholder="someone new…"
          aria-label="A new person's name" onChange={(e) => setValue(e.target.value)} onBlur={save}
          onKeyDown={(e) => { if (e.key === 'Enter') save(); if (e.key === 'Escape') { closed.current = true; setEditing(false) } }} />
      </span>
    )
  }
  // a named line: say it was really someone else (one click), or rename this person (the field)
  const others = segId ? ['You', ...known.filter((k) => k !== label)] : []
  return (
    <span className="who-pick" role="group" aria-label="Correct who said this">
      {others.map((k) => (
        <button key={k} type="button" className="who-chip" onMouseDown={(e) => e.preventDefault()}
          onClick={() => { closed.current = true; onAssign(segId!, k === 'You' ? 'me' : k); setEditing(false) }}>{k}</button>
      ))}
      <input className="who-input who-new" autoFocus value={value} maxLength={30} aria-label="Name for this person"
        title="Rename this person everywhere"
        onChange={(e) => setValue(e.target.value)} onBlur={save}
        onKeyDown={(e) => { if (e.key === 'Enter') save(); if (e.key === 'Escape') { closed.current = true; setEditing(false) } }} />
    </span>
  )
}

function TranscriptPanel({ lines, partial, names, details, micOn, onRename, onAssign }: {
  lines: Line[]; partial: string; names: Record<string, string>; details: boolean; micOn: boolean
  onRename: (speaker: number, name: string) => void
  onAssign: (seg: string, name: string) => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const lastKey = lines.length ? lines[lines.length - 1].key : 0
  useLayoutEffect(() => {
    const el = ref.current
    if (el) el.scrollTop = el.scrollHeight
  }, [lastKey, details, micOn])
  return (
    <section className="card-panel transcript-panel">
      <div className="panel-head">
        <h2>Conversation</h2>
        <span className="panel-note">Who said what, live</span>
      </div>
      <div ref={ref} className="transcript">
        {lines.length === 0 && (
          <div className="empty big">
            <div className="empty-icon"><Icon name={micOn ? 'ear' : 'micOff'} /></div>
            <div className="empty-title">{micOn ? 'Waiting for someone to speak' : 'The microphone is off'}</div>
            <p>{micOn
              ? 'Everything the band hears appears here with who said it. Raise your wrist to give Dextra a command.'
              : 'Double-tap or close your fist to start listening again.'}</p>
          </div>
        )}
        {lines.map((m, i) => {
          const who = whoOf(m.speaker, m.is_wearer, names, m.name, m.guess)
          const toCast = isToCast(m.mode)
          const prev = i > 0 ? lines[i - 1] : null
          const cont = !!prev && !toCast && prev.speaker === m.speaker && !!prev.is_wearer === !!m.is_wearer && isToCast(prev.mode) === toCast
          return (
            <div key={m.key} className={`line ${who.kind}${toCast ? ' to-cast' : ''}${cont ? ' cont' : ''}`} data-c={who.color}>
              <div className="avatar" aria-hidden="true">{cont ? '' : who.initial}</div>
              <div className="line-body">
                {!cont && (
                  <div className="line-head">
                    <SpeakerName label={who.label} speaker={m.speaker} segId={m.id} onRename={onRename} onAssign={onAssign}
                      known={Object.values(names)} editable={!m.is_wearer} />
                    {toCast && <span className="to-cast-label">to Dextra</span>}
                  </div>
                )}
                <div className="text">{m.text}</div>
                {details && (
                  <div className="tags">
                    <span className="tag">{m.mode}</span>
                    <span className="tag">{POSE_NAME[m.pose] ?? m.pose}</span>
                    <span className="tag">wearer {Number(m.wearer ?? 0).toFixed(2)}</span>
                    {m.rms_db != null && <span className="tag">{m.rms_db} dB</span>}
                    <span className="tag">speaker {m.speaker}</span>
                  </div>
                )}
              </div>
            </div>
          )
        })}
        {micOn && partial && (
          <div className="line live" aria-live="off">
            <div className="avatar" aria-hidden="true"><span className="partial-dots"><i /><i /><i /></span></div>
            <div className="line-body"><div className="text">{partial}</div></div>
          </div>
        )}
      </div>
    </section>
  )
}

/* ---------- right: tasks ---------- */

function useNow(intervalMs = 1000) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), intervalMs)
    return () => clearInterval(id)
  }, [intervalMs])
  return now
}

const clock = (epochS: number) =>
  new Date(epochS * 1000).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })

function countdown(epochS: number, nowMs: number): string {
  const total = Math.max(0, Math.ceil(epochS - nowMs / 1000))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  const pad = (n: number) => String(n).padStart(2, '0')
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`
}

const KIND_ICON: Record<TaskKind, string> = { time: 'clock', repeat: 'repeat', activity: 'activity', heard: 'ear' }

function TasksPanel({ reminders }: { reminders: Reminder[] }) {
  const now = useNow()
  const rank = (r: Reminder) => (r.status === 'due' ? -1 : r.remind_at ?? Number.MAX_SAFE_INTEGER)
  const sorted = [...reminders].sort((a, b) => rank(a) - rank(b))
  return (
    <div className="card-panel tasks-panel">
      <div className="panel-head">
        <h2>Tasks</h2>
        <span className="panel-note">{sorted.length > 0 ? `${sorted.length} ${sorted.length === 1 ? 'task' : 'tasks'}` : ''}</span>
      </div>
      <div className="tasks">
        {sorted.length === 0 && (
          <div className="empty">
            Nothing scheduled. Raise your wrist and try <q>remind me in five minutes to stretch</q>.
          </div>
        )}
        {sorted.map((r) => {
          const due = r.status === 'due'
          const at = typeof r.remind_at === 'number' ? r.remind_at : null
          const kind: TaskKind = r.kind_of ?? (at != null ? 'time' : 'heard')
          const trigger = r.trigger_label || (at != null ? `at ${clock(at)}` : 'when triggered')
          return (
            <div key={r.id} className={`task${due ? ' due' : ''}`}>
              <div className="task-icon"><Icon name={KIND_ICON[kind] ?? 'clock'} /></div>
              <div className="task-main">
                <div className="task-what">
                  {r.action_label && <span className="task-action">{r.action_label}</span>}
                  {r.task}
                </div>
                <div className="task-trigger">{due ? 'now' : trigger}</div>
              </div>
              {due
                ? <div className="task-due">DUE</div>
                : at != null && <div className="task-count">{countdown(at, now)}</div>}
            </div>
          )
        })}
      </div>
    </div>
  )
}

/* ---------- right: activity ---------- */

const VOICE_TITLES: ReadonlySet<string> = new Set(['New voice', 'Learned a voice'])

function StatusBadge({ status }: { status: ActionEvent['status'] }) {
  if (status === 'working') return <span className="badge working"><span className="spinner" />Working</span>
  if (status === 'done') return <span className="badge done"><Icon name="check" />Done</span>
  return <span className="badge refused"><Icon name="x" />Couldn’t do it</span>
}

function ActivityPanel({ cards, names }: { cards: Card[]; names: Record<string, string> }) {
  return (
    <div className="card-panel activity-panel">
      <div className="panel-head">
        <h2>Activity</h2>
        <span className="panel-note">What Dextra did</span>
      </div>
      <div className="cards">
        {cards.length === 0 && (
          <div className="empty">
            Nothing yet. Raise your wrist to your mouth and ask Dextra for something. What it does shows up here.
          </div>
        )}
        {cards.map((c) => {
          if (c.type === 'clip') {
            const who = c.ev.speaker != null ? whoOf(c.ev.speaker, false, names).label : null
            return (
              <div key={c.key} className="card clip">
                <div className="card-top">
                  <span className="badge clip"><Icon name="quote" />Clip saved</span>
                  {who && <span className="card-meta">{who}</span>}
                </div>
                <div className="card-quote">{c.ev.text}</div>
              </div>
            )
          }
          const detail = c.ev.result || c.ev.reason
          if (VOICE_TITLES.has(c.ev.text)) {
            return (
              <div key={c.key} className="card voice">
                <div className="card-top">
                  <span className="badge voice"><Icon name="person" />{c.ev.text}</span>
                </div>
                {detail && <div className="card-result">{detail}</div>}
              </div>
            )
          }
          return (
            <div key={c.key} className={`card ${c.ev.status}`}>
              <div className="card-top">
                <StatusBadge status={c.ev.status} />
                {c.ev.mode && c.ev.mode !== 'command' && <span className="card-meta">{String(c.ev.mode).replace(/_/g, ' ')}</span>}
              </div>
              <div className="card-quote">{c.ev.text}</div>
              {detail && <div className="card-result">{detail}</div>}
            </div>
          )
        })}
      </div>
    </div>
  )
}

/* ---------- due banner ---------- */

const BANNER_MS = 12_000

function DueBanner({ banner, onDismiss }: { banner: Banner; onDismiss: (key: number) => void }) {
  useEffect(() => {
    const id = setTimeout(() => onDismiss(banner.key), BANNER_MS)
    return () => clearTimeout(id)
  }, [banner.key, onDismiss])
  return (
    <div
      className="banner"
      role="alert"
      onClick={() => onDismiss(banner.key)}
      title="Click to dismiss"
    >
      <div className="banner-k">Reminder</div>
      <div className="banner-task">{banner.task}</div>
      {banner.say && banner.say !== banner.task && <div className="banner-say">{banner.say}</div>}
      <div className="banner-hint">Tap the band or click to dismiss</div>
      <div className="banner-timer" style={{ animationDuration: `${BANNER_MS}ms` }} />
    </div>
  )
}
