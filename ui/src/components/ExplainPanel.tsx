import { motion } from "framer-motion";

import {
  actionLabel,
  explainStep,
  formatFieldLabel,
  getRevealDeltas,
  getRuleDeltas,
  rewardTone,
  revealText,
  stepDoneText
} from "../lib/presentation";
import type { TraceStep } from "../types";

interface ExplainPanelProps {
  step: TraceStep | null;
  speed: number;
}

export function ExplainPanel({ step, speed }: ExplainPanelProps) {
  if (!step) {
    return (
      <section className="panel explain-panel">
        <div className="panel-header">
          <h2>What Happened</h2>
        </div>
        <div className="empty-block">Load a trace to inspect decision flow.</div>
      </section>
    );
  }

  const revealDeltas = getRevealDeltas(step);
  const ruleDeltas = getRuleDeltas(step);
  const explanation = explainStep(step);
  const rewardClass = rewardTone(step.reward);
  const timeCost = step.time_left_before - step.time_left_after;
  const applicantMoved = step.applicant_idx_after !== step.applicant_idx_before;

  return (
    <section className="panel explain-panel">
      <div className="panel-header">
        <h2>What Happened</h2>
        <p className="panel-subtext">{stepDoneText(step.done)}</p>
      </div>

      <motion.div
        key={step.step_idx}
        className="explain-main"
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: Math.max(0.14, 0.2 / speed) }}
      >
        <p className="explain-headline">{explanation.headline}</p>
        <p className="explain-detail">{explanation.detail}</p>
      </motion.div>

      <div className="explain-chips">
        <span className="badge action">Action: {actionLabel(step)}</span>
        <span className={`badge reward ${rewardClass}`}>Reward: {step.reward.toFixed(2)}</span>
        <span className="badge">Step: {step.step_idx + 1}</span>
      </div>

      <div className="delta-columns">
        <article className="delta-card">
          <h3>Reveal Changes</h3>
          {revealDeltas.length === 0 ? (
            <p className="muted">No field changed in this step.</p>
          ) : (
            <ul>
              {revealDeltas.map((delta) => (
                <li key={delta.key}>
                  <strong>{formatFieldLabel(delta.key)}</strong> {revealText(delta.before)} to {revealText(delta.after)}
                </li>
              ))}
            </ul>
          )}
        </article>

        <article className="delta-card">
          <h3>Rule Changes</h3>
          {ruleDeltas.length === 0 ? (
            <p className="muted">
              {step.rule_update_event ? "Rule update event emitted, but active constraints stayed the same." : "No rule delta."}
            </p>
          ) : (
            <ul>
              {ruleDeltas.map((delta) => (
                <li key={`${delta.label}-${delta.after}`}>
                  <strong>{delta.label}</strong> {delta.before} to {delta.after}
                </li>
              ))}
            </ul>
          )}
        </article>
      </div>

      <div className="micro-insight">
        <span>
          Time Cost: <strong>{timeCost}</strong>
        </span>
        <span>
          Queue:{" "}
          <strong>
            {step.applicant_idx_before + 1} to {step.applicant_idx_after + 1}
          </strong>{" "}
          ({applicantMoved ? "advanced" : "same applicant"})
        </span>
      </div>
    </section>
  );
}
