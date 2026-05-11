import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo } from "react";

import { usePlaybackStore } from "./store/playback";
import type { ReplayTrace, TraceStep } from "./types";

const REVEAL_FIELD_ORDER = [
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
];

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

const DECISION_LABEL: Record<string, string> = {
  correct: "Correct",
  false_accept: "False Accept",
  false_reject: "False Reject",
  none: "No Decision"
};

function revealText(value: number): string {
  if (value === -1) {
    return "UNKNOWN";
  }
  if (value === 1) {
    return "TRUE";
  }
  return "FALSE";
}

function revealClass(value: number): string {
  if (value === -1) {
    return "reveal unknown";
  }
  return value === 1 ? "reveal good" : "reveal bad";
}

function decisionClass(decision: string): string {
  if (decision === "false_accept" || decision === "false_reject") {
    return "decision bad";
  }
  if (decision === "correct") {
    return "decision good";
  }
  return "decision neutral";
}

function currentStep(trace: ReplayTrace | null, index: number): TraceStep | null {
  if (!trace || trace.steps.length === 0) {
    return null;
  }
  if (index < 0 || index >= trace.steps.length) {
    return null;
  }
  return trace.steps[index];
}

export default function App() {
  const manifest = usePlaybackStore((s) => s.manifest);
  const trace = usePlaybackStore((s) => s.trace);
  const selectedFile = usePlaybackStore((s) => s.selectedFile);
  const currentStepIndex = usePlaybackStore((s) => s.currentStep);
  const isPlaying = usePlaybackStore((s) => s.isPlaying);
  const speed = usePlaybackStore((s) => s.speed);
  const loading = usePlaybackStore((s) => s.loading);
  const error = usePlaybackStore((s) => s.error);
  const loadManifest = usePlaybackStore((s) => s.loadManifest);
  const selectTrace = usePlaybackStore((s) => s.selectTrace);
  const stepForward = usePlaybackStore((s) => s.stepForward);
  const stepBackward = usePlaybackStore((s) => s.stepBackward);
  const setStep = usePlaybackStore((s) => s.setStep);
  const togglePlay = usePlaybackStore((s) => s.togglePlay);
  const restart = usePlaybackStore((s) => s.restart);
  const cycleSpeed = usePlaybackStore((s) => s.cycleSpeed);

  useEffect(() => {
    void loadManifest();
  }, [loadManifest]);

  useEffect(() => {
    if (!isPlaying || !trace) {
      return;
    }
    const intervalMs = Math.max(140, Math.round(800 / speed));
    const timer = window.setInterval(() => {
      stepForward();
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [isPlaying, speed, stepForward, trace]);

  const step = currentStep(trace, currentStepIndex);
  const stepCount = trace?.steps.length ?? 0;
  const progress = stepCount > 0 ? ((currentStepIndex + 1) / stepCount) * 100 : 0;

  const displayedRules = useMemo(() => {
    if (!step?.rules_after) {
      return null;
    }
    return step.rules_after;
  }, [step]);

  return (
    <div className="app-shell">
      <header className="top-bar">
        <h1>NotWithoutPapers Replay Console</h1>
        <div className="controls">
          <select
            value={selectedFile ?? ""}
            onChange={(event) => void selectTrace(event.target.value)}
            disabled={!manifest || manifest.traces.length === 0}
          >
            {manifest?.traces.map((entry) => (
              <option key={entry.file} value={entry.file}>
                {entry.file}
              </option>
            ))}
          </select>
          <button onClick={togglePlay} disabled={!trace}>
            {isPlaying ? "Pause" : "Play"}
          </button>
          <button onClick={stepBackward} disabled={!trace}>
            Prev
          </button>
          <button onClick={stepForward} disabled={!trace}>
            Next
          </button>
          <button onClick={restart} disabled={!trace}>
            Restart
          </button>
          <button onClick={cycleSpeed} disabled={!trace}>
            {speed.toFixed(1)}x
          </button>
        </div>
      </header>

      <div className="status-line">
        {loading && <span>Loading trace data...</span>}
        {!loading && error && <span className="error">{error}</span>}
        {!loading && !error && trace && (
          <span>
            Trace seed <strong>{trace.meta.seed}</strong> · step {currentStepIndex + 1}/{stepCount}
          </span>
        )}
      </div>

      <main className="grid-layout">
        <section className="panel applicant-panel">
          <h2>Applicant Window</h2>
          <AnimatePresence mode="wait">
            {step ? (
              <motion.div
                key={`${step.step_idx}-${step.applicant_idx_before}`}
                className="applicant-card"
                initial={{ x: -120, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                exit={{ x: 120, opacity: 0 }}
                transition={{ duration: 0.26 }}
              >
                <p className="country">{step.applicant_before?.country_name ?? "N/A"}</p>
                <p className="meta">
                  Queue index {step.applicant_idx_before} · Time {step.time_left_before} → {step.time_left_after}
                </p>
                <p className="action">
                  Action: <strong>{step.action_name}</strong>
                </p>
                <p className={`decision ${decisionClass(step.decision_result)}`}>
                  Decision: {DECISION_LABEL[step.decision_result]}
                </p>
              </motion.div>
            ) : (
              <div className="empty-card">No trace selected.</div>
            )}
          </AnimatePresence>

          <div className="reveal-grid">
            {step &&
              REVEAL_FIELD_ORDER.map((field) => {
                const value = step.revealed_after[field];
                return (
                  <motion.div
                    key={`${step.step_idx}-${field}`}
                    className={revealClass(value)}
                    initial={{ rotateY: 90, opacity: 0.1 }}
                    animate={{ rotateY: 0, opacity: 1 }}
                    transition={{ duration: 0.22 }}
                  >
                    <span>{FIELD_LABELS[field]}</span>
                    <strong>{revealText(value)}</strong>
                  </motion.div>
                );
              })}
          </div>
        </section>

        <section className="panel rules-panel">
          <h2>Rules Board</h2>
          {displayedRules ? (
            <>
              <motion.div
                className="rule-line"
                animate={{
                  backgroundColor: step?.rule_update_event ? "#593f2f" : "#2f261f"
                }}
                transition={{ duration: 0.35 }}
              >
                Permit required: <strong>{displayedRules.permit_required ? "YES" : "NO"}</strong>
              </motion.div>
              <div className="rule-line">
                Citizen ID required: <strong>{displayedRules.id_card_required_for_citizens ? "YES" : "NO"}</strong>
              </div>
              <div className="rule-line">
                Work pass required: <strong>{displayedRules.work_pass_required ? "YES" : "NO"}</strong>
              </div>
              <div className="countries">
                {Object.entries(displayedRules.allowed_countries_by_name).map(([country, allowed]) => (
                  <span key={country} className={allowed ? "country-tag allowed" : "country-tag blocked"}>
                    {country}
                  </span>
                ))}
              </div>
              <AnimatePresence>
                {step?.rule_update_event && (
                  <motion.div
                    className="rule-update"
                    initial={{ y: -6, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    exit={{ y: -6, opacity: 0 }}
                  >
                    Rule update: {step.rule_update_event}
                  </motion.div>
                )}
              </AnimatePresence>
            </>
          ) : (
            <div className="empty-card">Rules unavailable.</div>
          )}
        </section>

        <section className="panel metrics-panel">
          <h2>Episode Metrics</h2>
          {trace ? (
            <>
              <p>
                Model: <code>{trace.meta.model_path}</code>
              </p>
              <p>
                Total Reward: <strong>{trace.episode.total_reward.toFixed(3)}</strong>
              </p>
              <div className="metric-grid">
                {Object.entries(trace.episode.stats).map(([k, v]) => (
                  <div key={k} className="metric-item">
                    <span>{k}</span>
                    <strong>{typeof v === "number" ? v.toFixed(3) : String(v)}</strong>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="empty-card">Metrics unavailable.</div>
          )}
        </section>
      </main>

      <footer className="timeline">
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
        <div className="timeline-steps">
          {trace?.steps.map((s) => (
            <button
              key={s.step_idx}
              onClick={() => setStep(s.step_idx)}
              className={[
                "step-chip",
                s.action_type,
                s.step_idx === currentStepIndex ? "active" : "",
                s.done ? "done" : ""
              ].join(" ")}
              title={`#${s.step_idx} ${s.action_name} | reward ${s.reward.toFixed(2)}`}
            >
              <motion.span
                animate={s.step_idx === currentStepIndex ? { scale: [1, 1.08, 1] } : { scale: 1 }}
                transition={{ duration: 0.35, repeat: s.step_idx === currentStepIndex ? Infinity : 0 }}
              >
                {s.step_idx}
              </motion.span>
            </button>
          ))}
        </div>
      </footer>
    </div>
  );
}
