export type Pose = 'mouth' | 'ear' | 'rest_table' | 'side' | 'chest' | 'extended' | 'glance' | 'moving'
export type Mode = 'command' | 'ask' | 'ambient' | 'note' | 'for_them' | 'status'

interface Base { t: number }

export interface PoseEvent extends Base { kind: 'pose'; pose: Pose; mode: Mode; angle: number | null }
export interface LevelEvent extends Base { kind: 'level'; rms_db: number }
export interface PartialEvent extends Base { kind: 'partial'; text: string }
export interface SegmentEvent extends Base {
  kind: 'segment'; speaker: number; text: string; start: number; end: number
  pose: Pose; rms_db: number | null; wearer: number; mode: Mode; is_wearer?: boolean; voice_sim?: number
}
export interface ActionEvent extends Base {
  kind: 'action'; id?: string; status: 'done' | 'refused' | 'working'; mode: Mode; text: string; result?: string; reason?: string
}
export interface ClipEvent extends Base { kind: 'clip'; text: string; speaker: number | null }
export type ReminderStatus = 'scheduled' | 'due' | 'cancelled'
export interface ReminderEvent extends Base {
  kind: 'reminder'; status: ReminderStatus; id: string; task: string; remind_at: number; say?: string
}
export interface RemindersEvent extends Base {
  kind: 'reminders'; items: { id: string; task: string; remind_at: number; status: ReminderStatus }[]
}

export type CastEvent =
  | PoseEvent | LevelEvent | PartialEvent | SegmentEvent | ActionEvent | ClipEvent | ReminderEvent | RemindersEvent

export interface Reminder {
  id: string; task: string; remind_at: number; status: ReminderStatus
  dueSince?: number // ms timestamp when it went due (for highlight)
}

export type Card =
  | { key: number; type: 'action'; ev: ActionEvent }
  | { key: number; type: 'clip'; ev: ClipEvent }

export interface Banner { key: number; id: string; task: string; say?: string }

export type ConnState = 'connecting' | 'live' | 'down'
