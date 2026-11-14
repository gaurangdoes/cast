import { useEffect, useMemo, useRef, useState } from 'react'

/* A guided setup: one step at a time, one button per step. It talks to the same calibration
   server as the older page (#calibrate-advanced), which stays available for everything else. */

type Recording = { label: string; rep: number; reps: number; phase: string; left?: number }
type Report = {
  accuracy?: number; error?: string; training?: boolean
  per_label?: Record<string, { recall: number; false_alarms: number }>
}
type ArmVerdict = { word: string; share_above: number; line_mg: number; text: string }
type Lab = {
  recording: Recording | null
  report: Report | null
  pose_report?: Report | null
  live: boolean
  detections: Record<string, number>
  counts: Record<string, number>
  voice_counts?: Record<string, number>
  arm?: { recordings?: Record<string, unknown>; verdict?: ArmVerdict | null }
}
type CalState = { band: string; status: string; error: string | null; lab?: Lab }
type Live = { pose?: string | null; rms_db: number | null }

type Step = {
  id: string
  group: string
  title: string
  say: string            // what to do, in one or two short sentences
  detail?: string
  button: string
  optional?: boolean
  done: (lab: Lab) => boolean
  run: (send: (c: object) => void) => void
  recLabel?: string      // matches lab.recording.label while this step records
}

const n = (lab: Lab, key: string) => lab.counts?.[key] ?? 0
const v = (lab: Lab, key: string) => lab.voice_counts?.[key] ?? 0
const arm = (lab: Lab, key: string) => !!lab.arm?.recordings?.[key]

const STEPS: Step[] = [
  {
    id: 'raised', group: 'How you hold your arm', title: 'Wrist raised',
    say: 'Hold the band up to your mouth, as if talking into it.',
    detail: 'Keep changing it a little for the whole 20 seconds: closer, further, different angles, sitting, standing.',
    button: 'Record 20 seconds', recLabel: 'pose_raised',
    done: (l) => n(l, 'pose_raised') >= 1, run: (s) => s({ cmd: 'lab_record', label: 'pose_raised', reps: 20 }),
  },
  {
    id: 'rest', group: 'How you hold your arm', title: 'Resting',
    say: 'Rest your arm the ways you normally do.',
    detail: 'On the table, in your lap, arms crossed, holding your phone. Move between them during the 20 seconds.',
    button: 'Record 20 seconds', recLabel: 'pose_rest',
    done: (l) => n(l, 'pose_rest') >= 1, run: (s) => s({ cmd: 'lab_record', label: 'pose_rest', reps: 20 }),
  },
  {
    id: 'side', group: 'How you hold your arm', title: 'At your side',
    say: 'Let your arm hang at your side.',
    detail: 'Stand for some of it and sit for some of it.',
    button: 'Record 20 seconds', recLabel: 'pose_side',
    done: (l) => n(l, 'pose_side') >= 1, run: (s) => s({ cmd: 'lab_record', label: 'pose_side', reps: 20 }),
  },
  {
    id: 'move', group: 'The fist', title: 'Normal movement',
    say: 'Move around the way you normally would for a minute. No fists.',
    detail: 'Type, pick things up, gesture while you talk, walk a few steps. This teaches Cast what is not a fist.',
    button: 'Record 60 seconds', recLabel: 'none',
    done: (l) => n(l, 'none') >= 1, run: (s) => s({ cmd: 'lab_record', label: 'none', reps: 60 }),
  },
  {
    id: 'fist1', group: 'The fist', title: 'Fist, round 1',
    say: 'Each time you hear “now”, clench your fist twice, quickly.',
    detail: '15 times. Hold your arm the way you will when you use it in the demo.',
    button: 'Record 15 fists', recLabel: 'fist',
    done: (l) => n(l, 'fist') >= 1, run: (s) => s({ cmd: 'lab_record', label: 'fist', reps: 15 }),
  },
  {
    id: 'fist2', group: 'The fist', title: 'Fist, round 2',
    say: 'Same again, 15 more.',
    detail: 'Squeeze the same way as in round 1.',
    button: 'Record 15 fists', recLabel: 'fist',
    done: (l) => n(l, 'fist') >= 2, run: (s) => s({ cmd: 'lab_record', label: 'fist', reps: 15 }),
  },
  {
    id: 'train', group: 'The fist', title: 'Train',
    say: 'Cast learns your poses and your fist from what you just recorded.',
    detail: 'Takes a few seconds. The result shows how often it got each one right on examples it had not seen.',
    button: 'Train',
    done: (l) => l.report?.accuracy != null && l.pose_report?.accuracy != null,
    run: (s) => s({ cmd: 'lab_train' }),
  },
  {
    id: 'arm_silent', group: 'Can the band feel your voice?', title: 'Silence',
    say: 'Everyone stays quiet. Rest your arm on the table and keep still.',
    button: 'Record 15 seconds', recLabel: 'arm_silent',
    done: (l) => arm(l, 'silent'), run: (s) => s({ cmd: 'arm_record', label: 'silent' }),
  },
  {
    id: 'arm_wearer', group: 'Can the band feel your voice?', title: 'You talking',
    say: 'Talk the whole time, with your arm resting on the table. Not at your mouth.',
    button: 'Record 15 seconds', recLabel: 'arm_wearer',
    done: (l) => arm(l, 'wearer'), run: (s) => s({ cmd: 'arm_record', label: 'wearer' }),
  },
  {
    id: 'arm_other', group: 'Can the band feel your voice?', title: 'Someone else talking',
    say: 'You stay silent. Someone else talks loudly from about half a metre away.',
    button: 'Record 15 seconds', recLabel: 'arm_other',
    done: (l) => arm(l, 'other'), run: (s) => s({ cmd: 'arm_record', label: 'other' }),
  },
  {
    id: 'voice_me', group: 'Voices', title: 'Your voice, arm at your side', optional: true,
    say: 'Talk for 40 seconds with your arm hanging at your side.',
    detail: 'Cast has your voice at your mouth and resting. This adds the third way it hears you.',
    button: 'Record 40 seconds', recLabel: 'voice_wearer',
    done: (l) => v(l, 'wearer') >= 3, run: (s) => s({ cmd: 'voice_record', label: 'wearer', seconds: 40 }),
  },
  {
    id: 'voice_other', group: 'Voices', title: 'The other person, once more', optional: true,
    say: 'They talk for 40 seconds from where they would normally stand. You stay silent.',
    detail: 'Cast has one recording of them. A second makes their label steadier.',
    button: 'Record 40 seconds', recLabel: 'voice_other1',
    done: (l) => v(l, 'other1') >= 2, run: (s) => s({ cmd: 'voice_record', label: 'other1', seconds: 40 }),
  },
]

function wsUrl() {
  return new URLSearchParams(location.search).get('ws') || `ws://${location.hostname || 'localhost'}:8765`
}

export default function Setup() {
  const [state, setState] = useState<CalState | null>(null)
  const [live, setLive] = useState<Live | null>(null)
  const [up, setUp] = useState(false)
  const [picked, setPicked] = useState<string | null>(null)
  const ws = useRef<WebSocket | null>(null)

  useEffect(() => {
    let stop = false
    let timer: number | undefined
    const open = () => {
      const s = new WebSocket(wsUrl())
      ws.current = s
      s.onopen = () => setUp(true)
      s.onclose = () => { setUp(false); if (!stop) timer = window.setTimeout(open, 1500) }
      s.onmessage = (ev) => {
        const m = JSON.parse(ev.data)
        if (m.kind === 'cal_state') setState(m)
        else if (m.kind === 'cal_live') setLive(m)
      }
    }
    open()
    return () => { stop = true; window.clearTimeout(timer); ws.current?.close() }
  }, [])

  const send = (c: object) => { if (ws.current?.readyState === 1) ws.current.send(JSON.stringify(c)) }
  const lab = state?.lab
  const connected = state?.status === 'connected' && !!lab
  const firstOpen = useMemo(() => (lab ? STEPS.find((s) => !s.done(lab))?.id ?? 'finish' : null), [lab])
  const currentId = picked ?? firstOpen
  const step = STEPS.find((s) => s.id === currentId)
  const doneCount = lab ? STEPS.filter((s) => s.done(lab)).length : 0
  const rec = lab?.recording ?? null

  // when the step in hand is finished, move to the next one that is not
  useEffect(() => {
    if (picked && lab && !rec) {
      const s = STEPS.find((x) => x.id === picked)
      if (s && s.done(lab) && !s.optional) setPicked(null)
    }
  }, [lab, picked, rec])

  return (
    <div className="setup">
      <header className="setup-top">
        <div className="brand">Cast <span className="setup-sub">Setup</span></div>
        <div className="setup-top-right">
          {connected && <span className="setup-progress">{doneCount} of {STEPS.length} done</span>}
          <a className="setup-link" href="#calibrate-advanced">All calibration tools</a>
        </div>
      </header>

      {!up && <Message title="Waiting for the calibration server" body="Start it with: python -m cast.calibration <band number>" />}

      {up && state && !connected && (
        <section className="setup-card setup-center">
          <div className="setup-eyebrow">First</div>
          <h1>Connect {state.band}</h1>
          <ol className="setup-list">
            <li>Close anything else that is using the band.</li>
            <li>Double-tap the band. Its light blinks blue for a minute.</li>
            <li>Press Connect while it is blinking.</li>
          </ol>
          <button className="setup-btn" disabled={state.status === 'connecting'} onClick={() => send({ cmd: 'connect' })}>
            {state.status === 'connecting' ? 'Connecting…' : 'Connect'}
          </button>
          {state.error && <p className="setup-error">{state.error}</p>}
        </section>
      )}

      {connected && lab && (
        <div className="setup-body">
          <nav className="setup-nav" aria-label="Steps">
            {Array.from(new Set(STEPS.map((s) => s.group))).map((g) => (
              <div key={g} className="setup-group">
                <div className="setup-eyebrow">{g}</div>
                {STEPS.filter((s) => s.group === g).map((s) => {
                  const done = s.done(lab)
                  return (
                    <button key={s.id} type="button" disabled={!!rec}
                      className={`setup-navitem${s.id === currentId ? ' on' : ''}${done ? ' done' : ''}`}
                      onClick={() => setPicked(s.id)}>
                      <span className="setup-tick" aria-hidden="true">{done ? '✓' : ''}</span>
                      <span>{s.title}{s.optional ? <em> · optional</em> : null}</span>
                    </button>
                  )
                })}
              </div>
            ))}
            <button type="button" disabled={!!rec} className={`setup-navitem${currentId === 'finish' ? ' on' : ''}`} onClick={() => setPicked('finish')}>
              <span className="setup-tick" aria-hidden="true" /><span>Finish</span>
            </button>
          </nav>

          <main className="setup-main">
            {rec ? <RecordingView rec={rec} step={step} />
              : step ? <StepView step={step} lab={lab} send={send} onSkip={() => {
                  const i = STEPS.findIndex((s) => s.id === step.id)
                  setPicked(STEPS[i + 1]?.id ?? 'finish')
                }} />
              : <Finish lab={lab} live={live} send={send} />}
          </main>
        </div>
      )}
    </div>
  )
}

function Message({ title, body }: { title: string; body: string }) {
  return <section className="setup-card setup-center"><h1>{title}</h1><p className="setup-detail"><code>{body}</code></p></section>
}

function RecordingView({ rec, step }: { rec: Recording; step?: Step }) {
  const counted = rec.reps > 0                       // fists: a spoken "now" for each one
  const big = rec.phase === 'ready' ? 'Get ready'
    : counted ? (rec.phase === 'now' ? 'NOW' : `${rec.rep} of ${rec.reps}`)
    : `${rec.left ?? ''}`
  return (
    <section className={`setup-card setup-center setup-rec ${rec.phase === 'ready' ? '' : 'live'}`}>
      <div className="setup-eyebrow">{rec.phase === 'ready' ? 'Starting' : 'Recording'}</div>
      <h1>{step?.title ?? 'Recording'}</h1>
      <div className="setup-big">{big}{!counted && rec.phase !== 'ready' ? <small>s</small> : null}</div>
      <p className="setup-say">{step?.say}</p>
      {step?.detail && <p className="setup-detail">{step.detail}</p>}
    </section>
  )
}

function StepView({ step, lab, send, onSkip }: { step: Step; lab: Lab; send: (c: object) => void; onSkip: () => void }) {
  const done = step.done(lab)
  const training = step.id === 'train' && (lab.report?.training || false)
  return (
    <section className="setup-card">
      <div className="setup-eyebrow">{step.group}</div>
      <h1>{step.title}</h1>
      <p className="setup-say">{step.say}</p>
      {step.detail && <p className="setup-detail">{step.detail}</p>}
      {step.id === 'train' && <TrainResult lab={lab} />}
      {step.id === 'arm_other' && <ArmResult lab={lab} />}
      <div className="setup-actions">
        <button className="setup-btn" disabled={training} onClick={() => step.run(send)}>
          {training ? 'Training…' : done ? `${step.button} again` : step.button}
        </button>
        <button className="setup-btn ghost" onClick={onSkip}>{done ? 'Next' : step.optional ? 'Skip' : 'Skip for now'}</button>
      </div>
      {done && <p className="setup-ok">✓ Done. You can redo it or move on.</p>}
    </section>
  )
}

const pct = (x?: number) => (x == null ? '–' : `${Math.round(x * 100)}%`)

function TrainResult({ lab }: { lab: Lab }) {
  const g = lab.report, p = lab.pose_report
  if (g?.error) return <p className="setup-error">{g.error}</p>
  if (g?.accuracy == null && p?.accuracy == null) return null
  const rows: [string, string, boolean][] = []
  if (p?.per_label) for (const [k, r] of Object.entries(p.per_label)) rows.push([k === 'raised' ? 'Wrist raised' : k === 'rest' ? 'Resting' : k === 'side' ? 'At your side' : k, pct(r.recall), r.recall >= 0.85])
  if (g?.per_label?.fist) rows.push(['Fist', `${pct(g.per_label.fist.recall)} found · ${g.per_label.fist.false_alarms} false`, g.per_label.fist.recall >= 0.85])
  return (
    <div className="setup-result">
      {rows.map(([k, val, ok]) => (
        <div key={k} className={`setup-row ${ok ? 'ok' : 'warn'}`}><span>{k}</span><b>{val}</b></div>
      ))}
      <p className="setup-detail">Each number is how often Cast was right on examples it had not trained on.</p>
    </div>
  )
}

function ArmResult({ lab }: { lab: Lab }) {
  const verdict = lab.arm?.verdict
  if (!verdict) return null
  const word = verdict.word === 'usable' ? 'Yes, it can' : verdict.word === 'marginal' ? 'Only just' : 'No'
  return (
    <div className="setup-result">
      <div className={`setup-row ${verdict.word === 'usable' ? 'ok' : 'warn'}`}><span>Can the band feel your voice?</span><b>{word}</b></div>
      <p className="setup-detail">{Math.round(verdict.share_above * 100)}% of the time you were talking, the vibration was above the level that silence and the other person stay under. {verdict.text}</p>
    </div>
  )
}

function Finish({ lab, live, send }: { lab: Lab; live: Live | null; send: (c: object) => void }) {
  const left = STEPS.filter((s) => !s.optional && !s.done(lab))
  return (
    <section className="setup-card">
      <div className="setup-eyebrow">Finish</div>
      <h1>{left.length ? `${left.length} step${left.length > 1 ? 's' : ''} left` : 'All set'}</h1>
      {left.length > 0 && <p className="setup-detail">Still to do: {left.map((s) => s.title).join(', ')}.</p>}
      <TrainResult lab={lab} />
      <ArmResult lab={lab} />
      <p className="setup-say">Try it before you leave.</p>
      <p className="setup-detail">Start the live test, then raise your wrist, rest it, and make a fist. The readout below should follow you.</p>
      <div className="setup-actions">
        <button className={`setup-btn${lab.live ? ' ghost' : ''}`} onClick={() => send({ cmd: 'lab_live', on: !lab.live })}>
          {lab.live ? 'Stop the live test' : 'Start the live test'}
        </button>
      </div>
      {lab.live && (
        <div className="setup-livebox">
          <div><div className="setup-eyebrow">Your arm is</div><div className="setup-livebig">{(live?.pose ?? '…').replace('_', ' ')}</div></div>
          <div><div className="setup-eyebrow">Fists seen</div><div className="setup-livebig">{lab.detections?.fist ?? 0}</div></div>
        </div>
      )}
      <p className="setup-detail">When you are happy, tell Claude it is done. Cast is restarted from the terminal, not from this page.</p>
    </section>
  )
}
