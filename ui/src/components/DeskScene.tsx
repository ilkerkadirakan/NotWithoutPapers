import { AnimatePresence, motion } from "framer-motion";

import {
  REVEAL_FIELD_ORDER,
  actionLabel,
  decisionTone,
  formatDecision,
  formatFieldLabel,
  revealText,
  revealTone
} from "../lib/presentation";
import type { TraceStep } from "../types";

interface DeskSceneProps {
  step: TraceStep | null;
  speed: number;
}

function decisionStamp(step: TraceStep | null): "APPROVED" | "DENIED" | null {
  if (!step) {
    return null;
  }
  if (step.action_type === "approve") {
    return "APPROVED";
  }
  if (step.action_type === "deny") {
    return "DENIED";
  }
  return null;
}

export function DeskScene({ step, speed }: DeskSceneProps) {
  const stamp = decisionStamp(step);
  const animDuration = Math.max(0.14, Math.min(0.3, 0.24 / speed));
  const applicantKey = step
    ? `${step.applicant_idx_before}-${step.applicant_before?.country_name ?? "unknown"}`
    : "none";

  return (
    <section className="panel desk-panel">
      <div className="panel-header">
        <h2>Border Desk</h2>
        {step && (
          <p className="panel-subtext">
            Applicant {step.applicant_idx_before + 1} | Time {step.time_left_before} to {step.time_left_after}
          </p>
        )}
      </div>

      <div className="desk-scene">
        <AnimatePresence mode="wait">
          {step ? (
            <motion.article
              key={applicantKey}
              className="applicant-card"
              initial={{ x: -56, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: 56, opacity: 0 }}
              transition={{ duration: animDuration }}
            >
              <p className="card-label">Applicant Window</p>
              <p className="country">{step.applicant_before?.country_name ?? "Unknown"}</p>
              <p className="action-line">{actionLabel(step)}</p>
              <p className={`decision-chip ${decisionTone(step.decision_result)}`}>
                {formatDecision(step.decision_result)}
              </p>
            </motion.article>
          ) : (
            <div className="empty-block">No trace selected.</div>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {stamp && (
            <motion.div
              key={`${step?.step_idx ?? -1}-${stamp}`}
              className={`stamp ${stamp === "APPROVED" ? "approve" : "deny"}`}
              initial={{ opacity: 0, scale: 1.35, rotate: -10 }}
              animate={{ opacity: 1, scale: 1, rotate: -8, x: [0, -4, 4, -2, 0] }}
              exit={{ opacity: 0 }}
              transition={{ duration: animDuration + 0.08 }}
            >
              {stamp}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="doc-grid">
        {step &&
          REVEAL_FIELD_ORDER.map((field) => {
            const value = step.revealed_after[field];
            const tone = revealTone(value);
            return (
              <motion.div
                key={`${step.step_idx}-${field}`}
                className={`doc-chip ${tone}`}
                initial={{ rotateX: 86, opacity: 0.25 }}
                animate={{ rotateX: 0, opacity: 1 }}
                transition={{ duration: animDuration }}
              >
                <span>{formatFieldLabel(field)}</span>
                <strong>{revealText(value)}</strong>
              </motion.div>
            );
          })}
      </div>
    </section>
  );
}
