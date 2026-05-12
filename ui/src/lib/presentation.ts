import type { RulesSnapshot, TraceStep } from "../types";

export const REVEAL_FIELD_ORDER = [
  "country_allowed",
  "has_permit",
  "expiry_valid",
  "name_match",
  "has_id_card",
  "is_worker",
  "has_work_pass",
  "purpose_match",
  "seal_valid",
  "biometric_match"
] as const;

const FIELD_LABELS: Record<string, string> = {
  country_allowed: "Country Allowed",
  has_permit: "Has Permit",
  expiry_valid: "Expiry Valid",
  name_match: "Name Match",
  has_id_card: "ID Card",
  is_worker: "Worker",
  has_work_pass: "Work Pass",
  purpose_match: "Purpose Match",
  seal_valid: "Seal Valid",
  biometric_match: "Biometric Match"
};

const DECISION_LABELS: Record<TraceStep["decision_result"], string> = {
  correct: "Correct",
  false_accept: "False Accept",
  false_reject: "False Reject",
  none: "No Decision"
};

interface FieldDelta {
  key: string;
  label: string;
  before: number;
  after: number;
}

interface RuleDelta {
  label: string;
  before: string;
  after: string;
}

interface ExplainContent {
  headline: string;
  detail: string;
}

export function revealText(value: number): string {
  if (value === -1) {
    return "UNKNOWN";
  }
  if (value === 1) {
    return "TRUE";
  }
  return "FALSE";
}

export function revealTone(value: number): "unknown" | "good" | "bad" {
  if (value === -1) {
    return "unknown";
  }
  return value === 1 ? "good" : "bad";
}

export function formatFieldLabel(field: string): string {
  return FIELD_LABELS[field] ?? field;
}

export function formatDecision(decision: TraceStep["decision_result"]): string {
  return DECISION_LABELS[decision];
}

export function decisionTone(decision: TraceStep["decision_result"]): "good" | "bad" | "neutral" {
  if (decision === "correct") {
    return "good";
  }
  if (decision === "false_accept" || decision === "false_reject") {
    return "bad";
  }
  return "neutral";
}

function boolText(value: number): string {
  return value ? "ON" : "OFF";
}

function ruleCountrySet(rules: RulesSnapshot | null): Set<string> {
  if (!rules) {
    return new Set();
  }
  const names = Object.keys(rules.allowed_countries_by_name).filter(
    (name) => rules.allowed_countries_by_name[name]
  );
  return new Set(names);
}

export function getRevealDeltas(step: TraceStep): FieldDelta[] {
  return REVEAL_FIELD_ORDER.flatMap((field) => {
    const before = step.revealed_before[field];
    const after = step.revealed_after[field];
    if (before === after) {
      return [];
    }
    return [
      {
        key: field,
        label: formatFieldLabel(field),
        before,
        after
      }
    ];
  });
}

export function getRuleDeltas(step: TraceStep): RuleDelta[] {
  const before = step.rules_before;
  const after = step.rules_after;
  if (!before || !after) {
    return [];
  }

  const changes: RuleDelta[] = [];
  if (before.permit_required !== after.permit_required) {
    changes.push({
      label: "Permit Required",
      before: boolText(before.permit_required),
      after: boolText(after.permit_required)
    });
  }
  if (before.id_card_required_for_citizens !== after.id_card_required_for_citizens) {
    changes.push({
      label: "Citizen ID Required",
      before: boolText(before.id_card_required_for_citizens),
      after: boolText(after.id_card_required_for_citizens)
    });
  }
  if (before.work_pass_required !== after.work_pass_required) {
    changes.push({
      label: "Work Pass Required",
      before: boolText(before.work_pass_required),
      after: boolText(after.work_pass_required)
    });
  }

  const beforeCountries = ruleCountrySet(before);
  const afterCountries = ruleCountrySet(after);
  const removed = [...beforeCountries].filter((name) => !afterCountries.has(name));
  const added = [...afterCountries].filter((name) => !beforeCountries.has(name));

  if (added.length > 0) {
    changes.push({
      label: "Allowed Countries",
      before: "unchanged",
      after: `+ ${added.join(", ")}`
    });
  }
  if (removed.length > 0) {
    changes.push({
      label: "Blocked Countries",
      before: "unchanged",
      after: `+ ${removed.join(", ")}`
    });
  }

  return changes;
}

export function rewardTone(value: number): "positive" | "negative" | "neutral" {
  if (value > 0) {
    return "positive";
  }
  if (value < 0) {
    return "negative";
  }
  return "neutral";
}

export function stepDoneText(done: TraceStep["done"]): string {
  if (done === "terminated") {
    return "Episode terminated";
  }
  if (done === "truncated") {
    return "Episode truncated";
  }
  return "In progress";
}

export function actionLabel(step: TraceStep): string {
  if (step.action_type === "inspect") {
    return `Inspect - ${step.action_name.replace("INSPECT_", "").replace(/_/g, " ")}`;
  }
  return step.action_name.replace(/_/g, " ");
}

export function explainStep(step: TraceStep): ExplainContent {
  const deltas = getRevealDeltas(step);
  const ruleDeltas = getRuleDeltas(step);

  if (step.action_type === "inspect") {
    if (deltas.length > 0) {
      return {
        headline: `Inspection revealed ${deltas.length} field${deltas.length > 1 ? "s" : ""}.`,
        detail: "Information quality improved for the current applicant."
      };
    }
    return {
      headline: "Inspection gave no new information.",
      detail: "Likely re-inspect or already-known field."
    };
  }

  if (step.decision_result === "correct") {
    return {
      headline: "Decision matched legal outcome.",
      detail: "Policy and observed evidence were aligned."
    };
  }

  if (step.decision_result === "false_accept") {
    return {
      headline: "False accept occurred.",
      detail: "An illegal applicant was approved."
    };
  }

  if (step.decision_result === "false_reject") {
    return {
      headline: "False reject occurred.",
      detail: "A legal applicant was denied."
    };
  }

  if (step.rule_update_event || ruleDeltas.length > 0) {
    return {
      headline: "Rules changed during this step.",
      detail: "Check the rule board for active constraints."
    };
  }

  return {
    headline: "No major state transition.",
    detail: "Episode continues with current policy and evidence."
  };
}

export function asPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export type { ExplainContent, FieldDelta, RuleDelta };
