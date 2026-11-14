import { useEffect, useRef, useState } from 'react'

type Step = { id: string; group: string; title: string; instruction: string }
type Summary = { gravity?: number[]; rms_db_median?: number; samples?: number; accel_spread_g?: number }
type Analysis = {
  pairs: { a: string; b: string; deg: number }[]
  loudness: { pose: string; you: number | null; teammate: number | null; gap: number | null }[]
  cover_drop: number | null
}
type CalState = {
  band: string; status: 'idle' | 'connecting' | 'connected' | 'error' | 'disconnected'; error: string | null
  steps: Step[]; done: Record<string, Summary>; active: { id: string; phase: 'countdown' | 'recording'; left: number } | null
  analysis: Analysis
}
type Live = { gravity: number[] | null; gyro: number; rms_db: number | null; nearest: { pose: string; deg: number } | null }

function wsUrl() {
  const q = new URLSearchParams(location.search).get('ws')
  return q || `ws://${location.hostname || 'localhost'}:8765`
}

export default function Calibrate() {
  const [state, setState] = useState<CalState | null>(null)
  const [live, setLive] = useState<Live | null>(null)
  const [conn, setConn] = useState<'connecting' | 'live' | 'down'>('connecting')
  const ws = useRef<WebSocket | null>(null)

  useEffect(() => {
    let stop = false
    let timer: number | undefined
    const open = () => {
      const s = new WebSocket(wsUrl())
      ws.current = s
      s.onopen = () => setConn('live')
      s.onclose = () => { setConn('down'); if (!stop) timer = window.setTimeout(open, 1500) }
      s.onmessage = (ev) => {
        const m = JSON.parse(ev.data)
        if (m.kind === 'cal_state') setState(m)
        else if (m.kind === 'cal_live') setLive(m)
      }
    }
    open()
    return () => { stop = true; window.clearTimeout(timer); ws.current?.close() }
  }, [])

  const send = (cmd: object) => ws.current?.readyState === 1 && ws.current.send(JSON.stringify(cmd))
  const connected = state?.status === 'connected'
  const busy = !!state?.active
  const groups = state ? Array.from(new Set(state.steps.map((s) => s.group))) : []
  const doneCount = state ? state.steps.filter((s) => state.done[s.id]).length : 0

  return (
    <div className="cal">
      <header className="cal-head">
        <div>
          <div className="brand">Cast · Calibrate</div>
          <div className={`status ${conn === 'live' ? 'live' : 'down'}`}>
            {conn === 'live' ? 'calibration server connected' : 'waiting for calib_server.py…'}
          </div>
        </div>
        <div className="cal-progress">{state ? `${doneCount} / ${state.steps.length} steps recorded` : ''}</div>
      </header>

      {state && !connected && (
        <section className="cal-connect">
          <h2>Connect {state.band}</h2>
          <ol>
            <li>Close the kit's dashboard tab and any other program using the band.</li>
            <li>Double-tap the band. Its light blinks blue for 60 seconds.</li>
            <li>Press Connect while it blinks.</li>
          </ol>
          <button className="btn primary" disabled={state.status === 'connecting'} onClick={() => send({ cmd: 'connect' })}>
            {state.status === 'connecting' ? 'Connecting…' : 'Connect'}
          </button>
          {state.error && <p className="cal-error">{state.error}</p>}
          {state.status === 'error' && (
            <p className="cal-hint">Still not found? In System Settings → Privacy &amp; Security → Bluetooth, allow the app you started
              calib_server.py from (Terminal, iTerm, VS Code), then restart that app.</p>
          )}
        </section>
      )}

      {state && connected && (
        <div className="cal-grid">
          <section className="panel">
            <h2>Live from the band</h2>
            <LiveView live={live} />
          </section>

          <section className="panel cal-steps">
            {state.active && <ActiveStep state={state} />}
            {groups.map((g) => (
              <div key={g} className="cal-group">
                <h2>{g}</h2>
                {state.steps.filter((s) => s.group === g).map((s) => {
                  const d = state.done[s.id]
                  const isActive = state.active?.id === s.id
                  return (
                    <div key={s.id} className={`cal-step ${d ? 'done' : ''} ${isActive ? 'active' : ''}`}>
                      <div className="cal-step-main">
                        <div className="cal-step-title">{d ? '✓ ' : ''}{s.title}</div>
                        <div className="cal-step-ins">{s.instruction}</div>
                        {d && <div className="cal-step-res">{describe(d)}</div>}
                      </div>
                      <button className="btn" disabled={busy} onClick={() => send({ cmd: 'record', step: s.id })}>
                        {d ? 'Redo' : 'Record'}
                      </button>
                    </div>
                  )
                })}
              </div>
            ))}
          </section>

          <section className="panel">
            <h2>Check</h2>
            <AnalysisView a={state.analysis} />
            <p className="cal-hint">Saved to data/calibration.json after every step. When the poses are well apart and you are clearly
              louder than your teammate, stop this server (Ctrl+C) and start <code>app.py 45</code>.</p>
            <button className="btn danger" disabled={busy} onClick={() => send({ cmd: 'reset' })}>Start over</button>
          </section>
        </div>
      )}
    </div>
  )
}

function describe(d: Summary) {
  const parts: string[] = []
  if (d.gravity) parts.push(`gravity ${d.gravity.map((x) => x.toFixed(2)).join(', ')}`)
  if (d.rms_db_median != null) parts.push(`${d.rms_db_median} dB`)
  if (d.accel_spread_g != null && d.accel_spread_g > 0.05) parts.push('moved a lot, consider redoing')
  return parts.join(' · ')
}

function ActiveStep({ state }: { state: CalState }) {
  const a = state.active!
  const step = state.steps.find((s) => s.id === a.id)
  return (
    <div className={`cal-active ${a.phase}`}>
      <div className="cal-active-phase">{a.phase === 'countdown' ? 'Get ready' : 'Recording — hold it'}</div>
      <div className="cal-active-title">{step?.title}</div>
      <div className="cal-active-ins">{step?.instruction}</div>
      <div className="cal-active-num">{a.phase === 'countdown' ? a.left : `${Math.max(0, a.left).toFixed(1)}s`}</div>
    </div>
  )
}

function LiveView({ live }: { live: Live | null }) {
  const g = live?.gravity
  const levelPct = live?.rms_db != null ? Math.max(0, Math.min(100, ((52 - live.rms_db) / 40) * 100)) : 0
  return (
    <div className="cal-live">
      {['x', 'y', 'z'].map((axis, i) => (
        <div key={axis} className="axis">
          <span className="axis-k">{axis}</span>
          <div className="axis-bar"><div className="axis-mid" />
            <div className="axis-fill" style={{
              left: g && g[i] < 0 ? `${50 + g[i] * 50}%` : '50%',
              width: g ? `${Math.abs(g[i]) * 50}%` : '0%',
            }} />
          </div>
          <span className="axis-v">{g ? g[i].toFixed(2) : '–'}</span>
        </div>
      ))}
      <div className="cal-nearest">
        {live?.nearest ? <>Looks like <b>{live.nearest.pose.replace('_', ' ')}</b> ({live.nearest.deg}° away)</> : 'Record a pose to see live matching'}
      </div>
      <h2>Mic level</h2>
      <div className="meter"><div style={{ width: `${levelPct}%` }} /></div>
      <div className="axis-v">{live?.rms_db != null ? `${live.rms_db} dB (smaller = louder)` : '–'}</div>
      <div className="axis-v">turning {live ? live.gyro : 0}°/s</div>
    </div>
  )
}

function AnalysisView({ a }: { a: Analysis }) {
  return (
    <div className="cal-analysis">
      <div className="cal-sub">Closest poses (under 20° may get confused)</div>
      {a.pairs.length === 0 && <div className="empty">Record at least two poses.</div>}
      {a.pairs.map((p) => (
        <div key={p.a + p.b} className={`cal-row ${p.deg < 20 ? 'bad' : 'ok'}`}>
          <span>{p.a.replace('_', ' ')} ↔ {p.b.replace('_', ' ')}</span><b>{p.deg}°</b>
        </div>
      ))}
      <div className="cal-sub">You vs teammate (bigger gap = easier)</div>
      {a.loudness.map((l) => (
        <div key={l.pose} className={`cal-row ${l.gap == null ? '' : l.gap >= 6 ? 'ok' : 'bad'}`}>
          <span>{l.pose.replace('_', ' ')}</span>
          <b>{l.gap == null ? '–' : `${l.gap > 0 ? '+' : ''}${l.gap} dB`}</b>
        </div>
      ))}
      <div className="cal-sub">Covering the band</div>
      <div className={`cal-row ${a.cover_drop == null ? '' : a.cover_drop >= 6 ? 'ok' : 'bad'}`}>
        <span>level drop when covered</span><b>{a.cover_drop == null ? '–' : `${a.cover_drop} dB`}</b>
      </div>
    </div>
  )
}
