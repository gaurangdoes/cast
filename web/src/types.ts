export type Pose = 'mouth' | 'ear' | 'rest_table' | 'side' | 'chest' | 'extended' | 'glance' | 'moving'
export type Mode = 'command' | 'ask' | 'ambient' | 'note' | 'for_them' | 'status'

interface Base { t: number }

export interface PoseEvent extends Base { kind: 'pose'; pose: Pose; mode: Mode; angle: number | null }
export interface LevelEvent extends Base { kind: 'level'; rms_db: number }
export interface PartialEvent extends Base { kind: 'partial'; text: string }
export interface SegmentEvent extends Base {
  kind: 'segment'; speaker: number; text: string; start: number; end: number
  pose: Pose; rms_db: number | null; wearer: number; mode: Mode; is_wearer?: boolean; voice_sim?: number; name?: string | null; id?: string
  guess?: string | null
}
export interface ActionEvent extends Base {
  kind: 'action'; id?: string; status: 'done' | 'refused' | 'working'; mode: Mode; text: string; result?: string; reason?: string
}
export interface ClipEvent extends Base { kind: 'clip'; text: string; speaker: number | null }
export type ReminderStatus = 'scheduled' | 'due' | 'cancelled'
export type TaskKind = 'time' | 'repeat' | 'activity' | 'heard'
/** Optional task fields: how it triggers and what it will do. remind_at is null for non-time triggers. */
export interface TaskFields {
  id: string; task: string; remind_at: number | null; status: ReminderStatus
  trigger_label?: string; action_label?: string; kind_of?: TaskKind
}
export interface ReminderEvent extends Base, TaskFields { kind: 'reminder'; say?: string }
export interface RemindersEvent extends Base { kind: 'reminders'; items: TaskFields[] }

export type CastEvent =
  | PoseEvent | LevelEvent | PartialEvent | SegmentEvent | ActionEvent | ClipEvent | ReminderEvent | RemindersEvent
  | { kind: 'names'; t: number; names: Record<string, string> }
  | { kind: 'relabel'; t: number; seg: string; speaker: number; is_wearer: boolean; name?: string | null }
  | { kind: 'ignored'; t: number; count: number }
  | { kind: 'battery'; t: number; percent: number; charging: boolean }
  | { kind: 'labelling'; t: number; mode: 'cast' | 'chat' | null; until: number | null }
  | { kind: 'mic'; t: number; on: boolean } | { kind: 'gesture'; t: number; label: string; prob: number }

export interface Reminder extends TaskFields {
  dueSince?: number // ms timestamp when it went due (for highlight)
}

export type Card =
  | { key: number; type: 'action'; ev: ActionEvent }
  | { key: number; type: 'clip'; ev: ClipEvent }

export interface Banner { key: number; id: string; task: string; say?: string }

export type ConnState = 'connecting' | 'live' | 'down'
