export type DoneState = false | "terminated" | "truncated";

export interface RulesSnapshot {
  allowed_countries_mask: number[];
  allowed_countries_by_name: Record<string, number>;
  permit_required: number;
  id_card_required_for_citizens: number;
  work_pass_required: number;
}

export interface ApplicantSnapshot {
  country_idx: number;
  country_name: string;
}

export interface TraceStep {
  step_idx: number;
  time_left_before: number;
  time_left_after: number;
  applicant_idx_before: number;
  applicant_idx_after: number;
  action_id: number;
  action_name: string;
  action_type: "inspect" | "approve" | "deny" | "unknown";
  reward: number;
  revealed_before: Record<string, number>;
  revealed_after: Record<string, number>;
  rules_before: RulesSnapshot | null;
  rules_after: RulesSnapshot | null;
  rule_update_event: string | null;
  decision_result: "correct" | "false_accept" | "false_reject" | "none";
  done: DoneState;
  applicant_before: ApplicantSnapshot | null;
  applicant_after: ApplicantSnapshot | null;
}

export interface TraceEpisode {
  episode_id: number;
  terminated: boolean;
  truncated: boolean;
  total_reward: number;
  stats: Record<string, number>;
}

export interface ReplayTrace {
  meta: {
    trace_version: string;
    model_path: string;
    seed: number;
    generated_at: string;
    env_config: Record<string, unknown>;
  };
  episode: TraceEpisode;
  steps: TraceStep[];
}

export interface TraceManifestEntry {
  id: string;
  file: string;
  seed: number;
  episode_id: number;
  model_path: string;
  total_reward: number;
  terminated: boolean;
  truncated: boolean;
}

export interface TracesManifest {
  trace_version: string;
  generated_at: string;
  traces: TraceManifestEntry[];
}
