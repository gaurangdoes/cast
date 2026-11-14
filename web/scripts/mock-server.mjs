// Throwaway mock of app.py's event stream for testing the UI without the band.
// Usage: pnpm mock   (listens on ws://localhost:8765)
import { WebSocketServer } from 'ws'

const PORT = Number(process.env.PORT || 8765)
const wss = new WebSocketServer({ port: PORT })
const now = () => Date.now() / 1000
const send = (ws, m) => ws.readyState === 1 && ws.send(JSON.stringify({ t: now(), ...m }))

const POSES = [
  ['rest_table', 'ambient'], ['mouth', 'command'], ['ear', 'ask'], ['side', 'ambient'],
  ['chest', 'note'], ['extended', 'for_them'], ['glance', 'status'], ['moving', 'ambient'],
]
const LINES = [
  [0, 0.92, 'mouth', 'command', 'Text Priya that I am running ten minutes late.'],
  [1, 0.12, 'rest_table', 'ambient', 'No worries, we can push the standup a bit.'],
  [0, 0.88, 'ear', 'ask', 'What is on my calendar after lunch?'],
  [2, 0.05, 'side', 'ambient', 'Did anyone grab the slides from last week?'],
  [0, 0.81, 'mouth', 'command', 'Remind me in two minutes to stretch.'],
]

wss.on('connection', (ws) => {
  console.log('client connected')
  const t0 = now()
  const reminders = [
    { id: 'r1', task: 'Check the oven', remind_at: t0 + 15, status: 'scheduled' },
    { id: 'r2', task: 'Call mom back', remind_at: t0 + 125, status: 'scheduled' },
    { id: 'r3', task: 'Submit the hackathon demo', remind_at: t0 + 3 * 3600 + 42, status: 'scheduled' },
    { id: 'r4', task: 'Water the plants', remind_at: t0 + 40, status: 'scheduled' },
  ]
  send(ws, { kind: 'reminders', items: reminders })

  let i = 0
  const timers = []
  timers.push(setInterval(() => send(ws, { kind: 'level', rms_db: 52 - Math.random() * 40 }), 100))
  timers.push(setInterval(() => {
    const [pose, mode] = POSES[i % POSES.length]
    send(ws, { kind: 'pose', pose, mode, angle: i % 5 === 4 ? null : Math.round(Math.random() * 25) })
  }, 2500))
  timers.push(setInterval(() => {
    const [speaker, wearer, pose, mode, text] = LINES[i % LINES.length]
    const words = text.split(' ')
    send(ws, { kind: 'partial', text: words.slice(0, Math.ceil(words.length / 2)).join(' ') + '…' })
    setTimeout(() => {
      send(ws, { kind: 'segment', speaker, text, start: now() - 2, end: now(), pose, rms_db: i % 3 ? -28 : null, wearer, mode })
      if (mode === 'command') send(ws, i % 4 ? { kind: 'action', status: 'done', mode, text, result: 'Sent.' } : { kind: 'action', status: 'refused', mode, text, reason: 'Not sure who you meant.' })
      if (mode === 'ask') send(ws, { kind: 'action', status: 'working', mode, text })
      if (i % 4 === 3) send(ws, { kind: 'clip', text, speaker })
      i++
    }, 1200)
  }, 4000))

  timers.push(setTimeout(() => send(ws, { kind: 'reminder', ...reminders[0], status: 'due', say: 'Hey, check the oven now.' }), 15000))
  timers.push(setTimeout(() => send(ws, { kind: 'reminder', ...reminders[3], status: 'cancelled' }), 25000))
  timers.push(setTimeout(() => send(ws, { kind: 'reminder', id: 'r5', task: 'Grab coffee', remind_at: now() + 20, status: 'scheduled' }), 8000))
  timers.push(setTimeout(() => send(ws, { kind: 'reminder', id: 'r5', task: 'Grab coffee', remind_at: now(), status: 'due' }), 28000))

  ws.on('close', () => { timers.forEach((t) => { clearInterval(t); clearTimeout(t) }); console.log('client disconnected') })
})
console.log(`mock Cast server on ws://localhost:${PORT}`)
