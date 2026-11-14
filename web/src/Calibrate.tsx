import { useEffect, useRef, useState } from 'react'

type Step = { id: string; group: string; title: string; instruction: string }
type Summary = { gravity?: number[]; rms_db_median?: number; samples?: number; accel_spread_g?: number; hp_median_mg?: number; hp_p95_mg?: number; hp_max_mg?: number; rate?: number }
type Analysis = {
  pairs: { a: string; b: string; deg: number }[]
  loudness: { pose: string; you: number | null; teammate: number | null; gap: number | null }[]
  cover_drop: number | null
  probe?: Record<string, Summary>
}
type CalState = {
  band: string; status: 'idle' | 'connecting' | 'connected' | 'error' | 'disconnected'; error: string | null
  steps: Step[]; done: Record<string, Summary>; active: { id: string; phase: 'countdown' | 'recording'; left: number } | null
  analysis: Analysis
  lab?: Lab
}
type LabReport = { accuracy?: number; samples?: Record<string, number>; per_label?: Record<string, { recall: number; false_alarms: number }>; error?: string; training?: boolean }
type Lab = {
  recording: { label: string; rep: number; reps: number; phase: string; left?: number } | null
  report: LabReport | null; live: boolean; detections: Record<string, number>
  last: { label: string; prob: number; at: number } | null; counts: Record<string, number>
  pose_report?: LabReport | null
}
type Live = { gravity: number[] | null; gyro: number; rms_db: number | null; nearest: { pose: string; deg: number } | null; pose?: string | null; pose_raw?: [string, number] | null }

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

      {state && connected && state.lab && <GestureLab lab={state.lab} send={send} live={live} />}

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
  if (d.hp_p95_mg != null) parts.push(`vibration median ${d.hp_median_mg} · p95 ${d.hp_p95_mg} · max ${d.hp_max_mg} mg @ ${d.rate}/s`)
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
      <div className="cal-sub">Fist test: vibration p95 (mg), fist should stand out from still and moving</div>
      {['rest', 'fist', 'hold', 'move'].map((k) => {
        const v = a.probe?.[k]
        const rest = a.probe?.rest?.hp_p95_mg
        const ratio = v?.hp_p95_mg != null && rest ? v.hp_p95_mg / rest : null
        return (
          <div key={k} className={`cal-row ${k === 'fist' && ratio != null ? (ratio >= 2 ? 'ok' : 'bad') : ''}`}>
            <span>{k}</span><b>{v?.hp_p95_mg != null ? `${v.hp_p95_mg}${ratio && k !== 'rest' ? ` (${ratio.toFixed(1)}× still)` : ''}` : '–'}</b>
          </div>
        )
      })}
    </div>
  )
}

const POSES: { id: string; name: string; how: string }[] = [
  { id: 'raised', name: 'Wrist raised', how: 'Band up near your mouth, like talking into it' },
  { id: 'rest', name: 'Resting', how: 'Arm on the table or your lap' },
  { id: 'side', name: 'At your side', how: 'Arm hanging down, standing or sitting' },
  { id: 'point', name: 'Pointing', how: 'Arm out toward someone, palm down' },
]

const GESTURES: { id: string; name: string; how: string }[] = [
  { id: 'fist', name: 'Fist', how: 'Close your fist hard, then open it' },
  { id: 'thumbs_up', name: 'Thumbs up', how: 'Thumbs up, then bring your hand back' },
]

function GestureLab({ lab, send, live }: { lab: Lab; send: (c: object) => void; live: Live | null }) {
  const rec = lab.recording
  const busy = !!rec
  const rpt = lab.report
  const [flash, setFlash] = useState<string | null>(null)
  useEffect(() => {
    if (!lab.last) return
    setFlash(lab.last.label)
    const t = window.setTimeout(() => setFlash(null), 900)
    return () => window.clearTimeout(t)
  }, [lab.last?.at])
  return (
    <section className="lab">
      <div className="lab-head">
        <h2>Gesture lab</h2>
        <span className="cal-hint">Record each gesture 10 times, plus normal movement, then train and test live.</span>
      </div>

      {rec && (
        <div className={`lab-rec ${rec.phase}`}>
          <div className="lab-rec-name">{rec.label === 'none' ? 'Normal movement' : rec.label.startsWith('pose_') ? POSES.find((g) => 'pose_' + g.id === rec.label)?.name : GESTURES.find((g) => g.id === rec.label)?.name}</div>
          {rec.label === 'none' || rec.label.startsWith('pose_')
            ? <div className="lab-rec-big">{rec.left}s</div>
            : <div className="lab-rec-big">{rec.phase === 'ready' ? 'Get ready' : rec.phase === 'now' ? 'NOW' : `${rec.rep} / ${rec.reps}`}</div>}
          <div className="cal-hint">{rec.label === 'none' ? 'Talk with your hands, scratch, rest. No gestures.' : rec.label.startsWith('pose_') ? 'Hold the pose but keep varying it: angles, sit, stand.' : GESTURES.find((g) => g.id === rec.label)?.how}</div>
        </div>
      )}

      <div className="cal-sub">Poses: record 20 s each, changing it a little the whole time</div>
      <div className="lab-grid">
        {POSES.map((g) => (
          <div key={g.id} className="lab-card">
            <div className="cal-step-title">{g.name}</div>
            <div className="cal-step-ins">{g.how}</div>
            <div className="lab-count">{lab.counts['pose_' + g.id] ?? 0} recording(s) · {lab.pose_report?.per_label?.[g.id] ? `${Math.round(lab.pose_report.per_label[g.id].recall * 100)}% correct` : 'not trained'}</div>
            <button className="btn" disabled={busy || lab.live} onClick={() => send({ cmd: 'lab_record', label: 'pose_' + g.id, reps: 20 })}>Record 20 s</button>
          </div>
        ))}
      </div>
      <div className="cal-sub">Gestures: 10 reps each, on every spoken "now"</div>
      <div className="lab-grid">
        {GESTURES.map((g) => (
          <div key={g.id} className="lab-card">
            <div className="cal-step-title">{g.name}</div>
            <div className="cal-step-ins">{g.how}. On each spoken "now".</div>
            <div className="lab-count">{lab.counts[g.id] ?? 0} recording(s) · {rpt?.samples?.[g.id] ?? 0} reps used</div>
            <button className="btn" disabled={busy || lab.live} onClick={() => send({ cmd: 'lab_record', label: g.id, reps: 10 })}>Record 10 reps</button>
          </div>
        ))}
        <div className="lab-card">
          <div className="cal-step-title">Normal movement</div>
          <div className="cal-step-ins">30 s of talking with your hands, scratching, resting. Teaches it what is NOT a gesture.</div>
          <div className="lab-count">{lab.counts['none'] ?? 0} recording(s) · {rpt?.samples?.['none'] ?? 0} windows used</div>
          <button className="btn" disabled={busy || lab.live} onClick={() => send({ cmd: 'lab_record', label: 'none', reps: 30 })}>Record 30 s</button>
        </div>
        <div className="lab-card">
          <div className="cal-step-title">Train</div>
          {lab.pose_report?.accuracy != null && <div className="lab-count">poses: {Math.round(lab.pose_report.accuracy * 100)}% correct overall</div>}
          {rpt?.training ? <div className="cal-step-ins">Training…</div>
            : rpt?.error ? <div className="cal-error">{rpt.error}</div>
            : rpt?.accuracy != null ? (
              <div className="lab-report">
                <div className="lab-acc">{Math.round(rpt.accuracy * 100)}% correct</div>
                {Object.entries(rpt.per_label ?? {}).map(([k, v]) => (
                  <div key={k} className={`cal-row ${k !== 'none' && v.recall < 0.9 ? 'bad' : 'ok'}`}>
                    <span>{k.replace('_', ' ')}</span><b>{Math.round(v.recall * 100)}% found · {v.false_alarms} false</b>
                  </div>
                ))}
              </div>
            ) : <div className="cal-step-ins">Each rep is tested by a model trained without it.</div>}
          <button className="btn primary" disabled={busy || lab.live} onClick={() => send({ cmd: 'lab_train' })}>Train</button>
        </div>
      </div>

      <div className="lab-live">
        <button className={`btn ${lab.live ? 'danger' : 'primary'}`} disabled={busy} onClick={() => send({ cmd: 'lab_live', on: !lab.live })}>
          {lab.live ? 'Stop live test' : 'Start live test'}
        </button>
        {lab.live && <button className="btn" onClick={() => send({ cmd: 'lab_reset_counts' })}>Reset counters</button>}
        {lab.live && (
          <div className="lab-pose">
            <div className="lab-rec-name">Pose now</div>
            <div className="lab-rec-big">{(live?.pose ?? '–').replace('_', ' ')}</div>
            <div className="lab-count">{live?.pose_raw ? `model: ${live.pose_raw[0]} ${Math.round(live.pose_raw[1] * 100)}%` : ''}</div>
          </div>
        )}
        {lab.live && (
          <div className="lab-counters">
            {GESTURES.map((g) => (
              <div key={g.id} className={`lab-counter ${flash === g.id ? 'hit' : ''}`}>
                <div className="lab-counter-n">{lab.detections[g.id] ?? 0}</div>
                <div className="lab-counter-k">{g.name}</div>
              </div>
            ))}
            <div className="cal-hint">Do each gesture 10 times, then move normally for a while. Counters should read 10 / 10 and not climb on their own.</div>
          </div>
        )}
      </div>
    </section>
  )
}
