import { useCallback, useEffect, useReducer } from 'react'
import type { Banner, Card, CastEvent, ConnState, Mode, Pose, Reminder, SegmentEvent } from './types'

const MAX_LINES = 300
const MAX_CARDS = 100
const DUE_KEEP_MS = 90_000 // due reminders stay highlighted in the list this long

export interface Line extends SegmentEvent { key: number }

export interface CastState {
  conn: ConnState
  pose: Pose | null
  mode: Mode
  angle: number | null
  poseSeen: boolean
  levelPct: number
  wearer: number | null
  lines: Line[]
  partial: string
  cards: Card[]
  reminders: Reminder[]
  banner: Banner | null
}

const initial: CastState = {
  conn: 'connecting', pose: null, mode: 'ambient', angle: null, poseSeen: false, levelPct: 0,
  wearer: null, lines: [], partial: '', cards: [], reminders: [], banner: null,
}

type Action =
  | { type: 'conn'; conn: ConnState }
  | { type: 'event'; ev: CastEvent }
  | { type: 'dismiss'; key: number }
  | { type: 'prune' }

let seq = 0
const nextKey = () => ++seq

function reduce(s: CastState, a: Action): CastState {
  switch (a.type) {
    case 'conn': return { ...s, conn: a.conn }
    case 'dismiss': return s.banner?.key === a.key ? { ...s, banner: null } : s
    case 'prune': {
      const cutoff = Date.now() - DUE_KEEP_MS
      const kept = s.reminders.filter((r) => r.status !== 'due' || (r.dueSince ?? 0) > cutoff)
      return kept.length === s.reminders.length ? s : { ...s, reminders: kept }
    }
    case 'event': return applyEvent(s, a.ev)
  }
}

function applyEvent(s: CastState, m: CastEvent): CastState {
  switch (m.kind) {
    case 'pose':
      return { ...s, pose: m.pose, mode: m.mode, angle: m.angle, poseSeen: true }
    case 'level': {
      const pct = Math.max(0, Math.min(100, ((52 - m.rms_db) / 40) * 100))
      return { ...s, levelPct: pct }
    }
    case 'partial':
      return { ...s, partial: m.text }
    case 'segment': {
      const lines = [...s.lines, { ...m, key: nextKey() }].slice(-MAX_LINES)
      return { ...s, lines, partial: '', wearer: m.wearer }
    }
    case 'action':
      if (m.id && s.cards.some((c) => c.type === 'action' && c.ev.id === m.id)) {
        return { ...s, cards: s.cards.map((c) => (c.type === 'action' && c.ev.id === m.id ? { ...c, ev: m } : c)) }
      }
      return { ...s, cards: [{ key: nextKey(), type: 'action' as const, ev: m }, ...s.cards].slice(0, MAX_CARDS) }
    case 'clip':
      return { ...s, cards: [{ key: nextKey(), type: 'clip' as const, ev: m }, ...s.cards].slice(0, MAX_CARDS) }
    case 'reminder': {
      const rest = s.reminders.filter((r) => r.id !== m.id)
      if (m.status === 'cancelled') {
        return { ...s, reminders: rest, banner: s.banner?.id === m.id ? null : s.banner }
      }
      if (m.status === 'due') {
        const r: Reminder = { id: m.id, task: m.task, remind_at: m.remind_at, status: 'due', dueSince: Date.now() }
        return { ...s, reminders: [...rest, r], banner: { key: nextKey(), id: m.id, task: m.task, say: m.say } }
      }
      return { ...s, reminders: [...rest, { id: m.id, task: m.task, remind_at: m.remind_at, status: 'scheduled' }] }
    }
    case 'reminders': {
      const prev = new Map(s.reminders.map((r) => [r.id, r]))
      const reminders = m.items
        .filter((i) => i.status !== 'cancelled')
        .map((i) => ({ ...i, dueSince: i.status === 'due' ? (prev.get(i.id)?.dueSince ?? Date.now()) : undefined }))
      return { ...s, reminders }
    }
    default:
      return s
  }
}

function wsUrl(): string {
  const override = new URLSearchParams(location.search).get('ws')
  if (override) return override
  return `ws://${location.hostname || 'localhost'}:8765`
}

export function useCast() {
  const [state, dispatch] = useReducer(reduce, initial)

  useEffect(() => {
    let ws: WebSocket | null = null
    let retry: ReturnType<typeof setTimeout> | undefined
    let stopped = false

    const connect = () => {
      if (stopped) return
      ws = new WebSocket(wsUrl())
      ws.onopen = () => dispatch({ type: 'conn', conn: 'live' })
      ws.onclose = () => {
        if (stopped) return
        dispatch({ type: 'conn', conn: 'down' })
        retry = setTimeout(connect, 1500)
      }
      ws.onmessage = (ev) => {
        try {
          const m = JSON.parse(ev.data as string) as CastEvent
          if (m && typeof m === 'object' && 'kind' in m) dispatch({ type: 'event', ev: m })
        } catch {
          /* ignore malformed frames */
        }
      }
    }
    connect()
    const prune = setInterval(() => dispatch({ type: 'prune' }), 5000)

    return () => {
      stopped = true
      clearTimeout(retry)
      clearInterval(prune)
      ws?.close()
    }
  }, [])

  const dismissBanner = useCallback((key: number) => dispatch({ type: 'dismiss', key }), [])
  return { state, dismissBanner }
}
