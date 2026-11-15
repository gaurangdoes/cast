// Throwaway mock of app.py's event stream for testing the UI without the band.
// Usage: pnpm mock                      (listens on ws://localhost:8765)
//        PORT=8799 pnpm mock            then open http://localhost:5173/?ws=ws://localhost:8799
// Plays a ~75 s scripted scene on a loop that exercises every event kind the screen handles:
// pose, level, partial, segment, action, clip, reminder, reminders, names, mic, gesture, ignored.
import { WebSocketServer } from 'ws'

const PORT = Number(process.env.PORT || 8765)
const wss = new WebSocketServer({ port: PORT })
const now = () => Date.now() / 1000
const send = (ws, m) => ws.readyState === 1 && ws.send(JSON.stringify({ t: now(), ...m }))

const LOOP_S = 76

wss.on('connection', (ws) => {
  console.log('client connected')
  const timers = []
  let aid = 0
  let ignored = 0
  let micOn = true
  let loud = false // someone is speaking: push the level meter up

  const at = (s, fn) => timers.push(setTimeout(fn, s * 1000))
  const pose = (p, mode) => send(ws, { kind: 'pose', pose: p, mode, angle: Math.round(Math.random() * 20) })

  // One utterance: a partial first, then the final segment 1.2 s later.
  const say = (s, speaker, text, o = {}) => {
    const mode = o.mode ?? 'ambient'
    const words = text.split(' ')
    at(s, () => { loud = true; send(ws, { kind: 'partial', text: words.slice(0, Math.ceil(words.length / 2)).join(' ') + '…' }) })
    at(s + 1.2, () => {
      loud = false
      send(ws, {
        kind: 'segment', speaker, text, start: now() - 2, end: now(), mode,
        pose: o.pose ?? (mode === 'ambient' ? 'rest_table' : 'mouth'),
        rms_db: o.db ?? (o.you ? 21 : 33), wearer: o.you ? 0.9 : 0.1, is_wearer: !!o.you, name: o.name ?? null,
      })
    })
  }
  // A command card that goes working -> done / refused, updated in place by id.
  const command = (s, text, status, detail, delay = 2.5) => {
    const id = `a${++aid}`
    at(s, () => send(ws, { kind: 'action', id, status: 'working', mode: 'command', text }))
    at(s + delay, () => send(ws, { kind: 'action', id, status, mode: 'command', text, ...(status === 'done' ? { result: detail } : { reason: detail }) }))
  }
  const ignore = (s) => at(s, () => send(ws, { kind: 'ignored', count: ++ignored }))

  timers.push(setInterval(() => {
    const base = !micOn ? 50 : loud ? 24 : 42
    send(ws, { kind: 'level', rms_db: base + Math.random() * 8 })
  }, 100))

  const scene = () => {
    const t0 = now()
    const names = { 1: 'Hardik', 2: 'Person 2' }
    const tasks = [
      { id: 'r1', task: 'Stand up and stretch', remind_at: t0 + 22, status: 'scheduled', kind_of: 'time', action_label: 'Say', trigger_label: 'in 22 seconds' },
      { id: 'r2', task: 'Drink some water', remind_at: t0 + 20 * 60, status: 'scheduled', kind_of: 'repeat', action_label: 'Say', trigger_label: 'every 20 min' },
      { id: 'r3', task: 'Recap what you missed', remind_at: null, status: 'scheduled', kind_of: 'activity', action_label: 'Recap', trigger_label: 'when you get up' },
      { id: 'r4', task: 'Flag the submission deadline', remind_at: null, status: 'scheduled', kind_of: 'heard', action_label: 'Say', trigger_label: 'when Hardik mentions the deadline' },
      { id: 'r5', task: 'Submit the hackathon demo', remind_at: t0 + 3 * 3600 + 42, status: 'scheduled' }, // legacy shape: no optional fields
    ]
    micOn = true; aid = 0
    send(ws, { kind: 'mic', on: true })
    send(ws, { kind: 'names', names: { 1: 'Person 1' } })
    send(ws, { kind: 'reminders', items: tasks })
    pose('rest_table', 'ambient')

    say(1.5, 1, 'Okay, are we ready to run through the demo once more?')
    at(3, () => send(ws, { kind: 'action', id: 'v1', status: 'done', mode: 'command', text: 'New voice', result: 'Hearing someone new: calling them Person 1 for now.' }))
    say(5, 0, 'Almost. Give me two minutes to check the slides.', { you: true })
    ignore(6.5)
    say(8.5, -1, 'Is this seat taken?', { db: 44 })
    ignore(10)

    // Raise wrist: command that succeeds, and teaches a name
    at(11.5, () => pose('mouth', 'command'))
    say(12, 0, 'This is Hardik.', { you: true, mode: 'command' })
    at(14, () => { send(ws, { kind: 'names', names }); send(ws, { kind: 'action', id: 'v2', status: 'done', mode: 'command', text: 'Learned a voice', result: 'Speaker 1 is Hardik (you introduced them)' }) })
    say(15.5, 0, 'Remind me in ten minutes to email the judges.', { you: true, mode: 'command' })
    command(17, 'Remind me in ten minutes to email the judges.', 'done', 'Okay. I will remind you at ' + new Date((t0 + 617) * 1000).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) + '.')
    at(19.5, () => send(ws, { kind: 'reminder', id: 'r6', task: 'Email the judges', remind_at: now() + 600, status: 'scheduled', kind_of: 'time', action_label: 'Say', trigger_label: 'in 10 min' }))
    at(19.5, () => pose('rest_table', 'ambient'))

    say(21, 1, 'The deadline moved to four, by the way.', { name: 'Hardik' })
    at(22, () => send(ws, { kind: 'reminder', ...tasks[0], status: 'due', say: 'Time to stand up and stretch.' }))
    say(24.5, 2, 'Four? I thought it was five thirty.')
    say(27.5, 1, 'They sent an update this morning.', { name: 'Hardik' })
    at(29, () => send(ws, { kind: 'clip', text: 'The deadline moved to four, by the way.', speaker: 1 }))
    ignore(30); ignore(30.5)

    // Fist gesture: mic pause / resume
    at(33, () => { send(ws, { kind: 'gesture', label: 'fist', prob: 0.94 }); micOn = false; send(ws, { kind: 'mic', on: false }) })
    at(39, () => { send(ws, { kind: 'gesture', label: 'double_tap', prob: 0.9 }); micOn = true; send(ws, { kind: 'mic', on: true }) })

    // Raise wrist: a question, then a command that gets refused
    at(41, () => pose('ear', 'ask'))
    say(41.5, 0, 'What did Hardik say about the deadline?', { you: true, mode: 'ask', pose: 'ear' })
    command(43, 'What did Hardik say about the deadline?', 'done', 'Hardik said the deadline moved to four. An update went out this morning.', 3)
    at(47, () => pose('mouth', 'command'))
    say(47.5, 0, 'Text everyone in the room that we are live.', { you: true, mode: 'command' })
    command(49, 'Text everyone in the room that we are live.', 'refused', 'I can only message people you have named. Tell me who to text.')
    at(52, () => pose('side', 'ambient'))

    say(53.5, 2, 'Can we go over who presents which part?')
    say(56.5, 0, 'I will take the intro and the live demo.', { you: true })
    say(59, 0, 'Hardik covers the architecture.', { you: true })
    ignore(60)
    say(62, -1, 'Testing, one, two.', { db: 46 })
    at(63, () => send(ws, { kind: 'reminder', ...tasks[4], status: 'cancelled' }))
    at(64, () => send(ws, { kind: 'gesture', label: 'thumbs_up', prob: 0.88 }))
    say(66, 1, 'Sounds good. Let us do it.', { name: 'Hardik' })
    at(68, () => send(ws, { kind: 'reminder', ...tasks[2], status: 'due', say: 'While you were away: the deadline moved to four.' }))
    at(71, () => send(ws, { kind: 'reminders', items: tasks.filter((x) => x.id === 'r2' || x.id === 'r4') }))

    at(LOOP_S, scene)
  }
  scene()

  ws.on('close', () => { timers.forEach((t) => { clearInterval(t); clearTimeout(t) }); console.log('client disconnected') })
})
console.log(`mock Dextra server on ws://localhost:${PORT}`)
