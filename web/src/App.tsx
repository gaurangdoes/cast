import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { useCast, type Line } from './useCast'
import type { Banner, Card, ConnState, Mode, Reminder } from './types'

const MODE_COLOR: Record<string, string> = {
  command: 'var(--command)', ask: 'var(--ask)', ambient: 'var(--ambient)',
  note: 'var(--note)', for_them: 'var(--for_them)', status: 'var(--status)',
}
const POSE_NAME: Record<string, string> = {
  mouth: 'Wrist to mouth', ear: 'Wrist raised', rest_table: 'Resting', side: 'At your side',
  chest: 'On chest', extended: 'Arm out', glance: 'Glance', moving: 'Moving',
}
const spkColor = (s: number | null | undefined) => `var(--spk${(s ?? 0) % 4})`
const modeColor = (m: string | undefined) => (m && MODE_COLOR[m]) || 'var(--muted)'

export default function App() {
  const { state, dismissBanner } = useCast()
  return (
    <>
      {state.banner && <DueBanner key={state.banner.key} banner={state.banner} onDismiss={dismissBanner} />}
      <div className="app">
        <WristPanel
          conn={state.conn} pose={state.pose} mode={state.mode} angle={state.angle}
          poseSeen={state.poseSeen} levelPct={state.levelPct} wearer={state.wearer}
        />
        <TranscriptPanel lines={state.lines} partial={state.partial} />
        <section className="panel right">
          <RemindersPanel reminders={state.reminders} />
          <CardsPanel cards={state.cards} />
        </section>
      </div>
    </>
  )
}

function ConnStatus({ conn }: { conn: ConnState }) {
  const label = conn === 'live' ? 'live' : conn === 'connecting' ? 'connecting…' : 'disconnected, retrying…'
  return <div className={`status ${conn === 'live' ? 'live' : 'down'}`}>{label}</div>
}

function WristPanel(p: {
  conn: ConnState; pose: string | null; mode: Mode; angle: number | null
  poseSeen: boolean; levelPct: number; wearer: number | null
}) {
  return (
    <section className="panel">
      <div>
        <div className="brand">Cast</div>
        <ConnStatus conn={p.conn} />
      </div>
      <h2>Wrist</h2>
      <div className="pose">{p.pose ? (POSE_NAME[p.pose] ?? p.pose) : '—'}</div>
      <span className="mode" style={{ background: MODE_COLOR[p.mode] ?? 'var(--ambient)' }}>{p.mode}</span>
      <div className="angle">
        {p.poseSeen ? (p.angle != null ? `${p.angle}° from nearest calibrated pose` : 'not calibrated') : ''}
      </div>
      <h2>Mic level</h2>
      <div className="meter"><div style={{ width: `${p.levelPct}%` }} /></div>
      <h2>Last speaker is the wearer</h2>
      <div className="wearer-big">{p.wearer != null ? Number(p.wearer).toFixed(2) : '–'}</div>
      <div className="legend">
        <b>wrist to mouth</b><span>talk to Cast: commands and questions</span>
        <b>resting</b><span>ambient: remembered only</span>
        <b>double-tap</b><span>clip what was just said</span>
        <b>glance</b><span>hear your next reminder</span>
      </div>
    </section>
  )
}

function TranscriptPanel({ lines, partial }: { lines: Line[]; partial: string }) {
  const ref = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    const el = ref.current
    if (el) el.scrollTop = el.scrollHeight
  }, [lines.length])
  return (
    <section className="panel">
      <h2>Transcript · who said what</h2>
      <div ref={ref} className="transcript">
        {lines.length === 0 && <div className="empty">Waiting for speech…</div>}
        {lines.map((m) => {
          const who = m.is_wearer ? `You` : `Speaker ${m.speaker}`
          return (
            <div key={m.key} className={`line ${m.mode ?? ''}`}>
              <div className="who" style={{ color: spkColor(m.speaker) }}>{who}</div>
              <div>
                <div className="text">{m.text}</div>
                <div className="tags">
                  <span className="tag" style={{ color: modeColor(m.mode) }}>{m.mode}</span>
                  <span className="tag">{POSE_NAME[m.pose] ?? m.pose}</span>
                  <span className="tag">wearer {Number(m.wearer).toFixed(2)}</span>
                  {m.rms_db != null && <span className="tag">{m.rms_db} dB</span>}
                </div>
              </div>
            </div>
          )
        })}
      </div>
      <div className="partial">{partial}</div>
    </section>
  )
}

function CardsPanel({ cards }: { cards: Card[] }) {
  return (
    <div className="section grow">
      <h2>Actions and clips</h2>
      <div className="cards">
        {cards.length === 0 && <div className="empty">Raise your wrist to your mouth and give a command.</div>}
        {cards.map((c) =>
          c.type === 'action' ? (
            <div key={c.key} className={`card ${c.ev.status}`}>
              <div className="k">{c.ev.mode} · {c.ev.status}</div>
              <div className="q">{c.ev.text}</div>
              {(c.ev.result || c.ev.reason) && <div className="r">{c.ev.result || c.ev.reason}</div>}
            </div>
          ) : (
            <div key={c.key} className="card clip">
              <div className="k">{c.ev.speaker != null ? `clip · speaker ${c.ev.speaker}` : 'clip'}</div>
              <div className="q">{c.ev.text}</div>
            </div>
          ),
        )}
      </div>
    </div>
  )
}

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

function RemindersPanel({ reminders }: { reminders: Reminder[] }) {
  const now = useNow()
  const sorted = [...reminders].sort((a, b) => a.remind_at - b.remind_at)
  return (
    <div className="section">
      <h2>Reminders</h2>
      <div className="reminders">
        {sorted.length === 0 && <div className="empty">No reminders. Try “remind me in 5 minutes to…”</div>}
        {sorted.map((r) => {
          const due = r.status === 'due'
          return (
            <div key={r.id} className={`reminder${due ? ' due' : ''}`}>
              <div className="rem-task">{r.task}</div>
              <div className="rem-meta">
                <span className="rem-clock">{clock(r.remind_at)}</span>
                <span className="rem-count">{due ? 'DUE NOW' : countdown(r.remind_at, now)}</span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

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
      <div className="banner-hint">click to dismiss</div>
      <div className="banner-timer" style={{ animationDuration: `${BANNER_MS}ms` }} />
    </div>
  )
}
