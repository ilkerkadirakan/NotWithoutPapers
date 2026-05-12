import { REVEAL_FIELD_ORDER, formatFieldLabel } from "./presentation";
import type { ReplayTrace, TraceStep } from "../types";

type StoryDecision = TraceStep["decision_result"] | "pending";

export interface ApplicantStory {
  applicantIndex: number;
  countryName: string;
  startStep: number;
  endStep: number;
  decisionStep: number;
  decision: StoryDecision;
  decisionAction: "approve" | "deny" | "unknown";
  inspectCount: number;
  rewardTotal: number;
  timeCost: number;
  revealedFields: string[];
  finalKnownFields: string[];
  ruleUpdates: string[];
}

interface StoryBuildState {
  applicantIndex: number;
  countryName: string;
  startStep: number;
  endStep: number;
  decisionStep: number;
  decision: StoryDecision;
  decisionAction: "approve" | "deny" | "unknown";
  inspectCount: number;
  rewardTotal: number;
  timeCost: number;
  revealedSet: Set<string>;
  finalKnownSet: Set<string>;
  ruleUpdates: string[];
}

function newStory(step: TraceStep): StoryBuildState {
  return {
    applicantIndex: step.applicant_idx_before,
    countryName: step.applicant_before?.country_name ?? "Unknown",
    startStep: step.step_idx,
    endStep: step.step_idx,
    decisionStep: step.step_idx,
    decision: "pending",
    decisionAction: "unknown",
    inspectCount: 0,
    rewardTotal: 0,
    timeCost: 0,
    revealedSet: new Set<string>(),
    finalKnownSet: new Set<string>(),
    ruleUpdates: []
  };
}

function absorbStep(state: StoryBuildState, step: TraceStep): void {
  state.endStep = step.step_idx;
  state.rewardTotal += step.reward;
  state.timeCost += Math.max(0, step.time_left_before - step.time_left_after);

  if (step.action_type === "inspect") {
    state.inspectCount += 1;
  }
  if (step.action_type === "approve") {
    state.decisionAction = "approve";
  } else if (step.action_type === "deny") {
    state.decisionAction = "deny";
  }

  if (step.decision_result !== "none") {
    state.decision = step.decision_result;
    state.decisionStep = step.step_idx;
    if (step.decision_result === "false_accept") {
      state.decisionAction = "approve";
    }
    if (step.decision_result === "false_reject") {
      state.decisionAction = "deny";
    }
  }

  for (const field of REVEAL_FIELD_ORDER) {
    const before = step.revealed_before[field];
    const after = step.revealed_after[field];
    if (before === -1 && after !== -1) {
      state.revealedSet.add(formatFieldLabel(field));
    }
    if (after !== -1) {
      state.finalKnownSet.add(formatFieldLabel(field));
    }
  }

  if (step.rule_update_event && !state.ruleUpdates.includes(step.rule_update_event)) {
    state.ruleUpdates.push(step.rule_update_event);
  }
}

function finalizeStory(state: StoryBuildState): ApplicantStory {
  return {
    applicantIndex: state.applicantIndex,
    countryName: state.countryName,
    startStep: state.startStep,
    endStep: state.endStep,
    decisionStep: state.decisionStep,
    decision: state.decision,
    decisionAction: state.decisionAction,
    inspectCount: state.inspectCount,
    rewardTotal: state.rewardTotal,
    timeCost: state.timeCost,
    revealedFields: [...state.revealedSet],
    finalKnownFields: [...state.finalKnownSet],
    ruleUpdates: state.ruleUpdates
  };
}

function applicantFinished(step: TraceStep): boolean {
  if (step.decision_result !== "none") {
    return true;
  }
  return step.applicant_idx_after > step.applicant_idx_before;
}

export function buildApplicantStories(trace: ReplayTrace | null): ApplicantStory[] {
  if (!trace || trace.steps.length === 0) {
    return [];
  }

  const stories: ApplicantStory[] = [];
  let current: StoryBuildState | null = null;

  for (const step of trace.steps) {
    if (!current || current.applicantIndex !== step.applicant_idx_before) {
      if (current) {
        stories.push(finalizeStory(current));
      }
      current = newStory(step);
    }

    absorbStep(current, step);

    if (applicantFinished(step)) {
      stories.push(finalizeStory(current));
      current = null;
    }
  }

  if (current) {
    stories.push(finalizeStory(current));
  }

  return stories;
}

export function storyDecisionLabel(story: ApplicantStory): string {
  if (story.decision === "correct") {
    if (story.decisionAction === "approve") {
      return "Approved (correct)";
    }
    if (story.decisionAction === "deny") {
      return "Denied (correct)";
    }
    return "Correct decision";
  }
  if (story.decision === "false_accept") {
    return "False accept";
  }
  if (story.decision === "false_reject") {
    return "False reject";
  }
  return "No final decision";
}

export function storyTone(story: ApplicantStory): "good" | "bad" | "neutral" {
  if (story.decision === "correct") {
    return "good";
  }
  if (story.decision === "false_accept" || story.decision === "false_reject") {
    return "bad";
  }
  return "neutral";
}
