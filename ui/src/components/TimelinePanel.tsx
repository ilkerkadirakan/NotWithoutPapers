import { motion } from "framer-motion";

import type { ReplayTrace } from "../types";

interface TimelinePanelProps {
  trace: ReplayTrace | null;
  currentStep: number;
  setStep: (step: number) => void;
}

function actionTone(actionType: string): "inspect" | "approve" | "deny" | "unknown" {
  if (actionType === "inspect" || actionType === "approve" || actionType === "deny") {
    return actionType;
  }
  return "unknown";
}

export function TimelinePanel({ trace, currentStep, setStep }: TimelinePanelProps) {
  const stepCount = trace?.steps.length ?? 0;
  const maxIdx = Math.max(0, stepCount - 1);
  const progress = stepCount > 1 ? (currentStep / maxIdx) * 100 : stepCount === 1 ? 100 : 0;

  return (
    <footer className="panel timeline-panel">
      <div className="timeline-head">
        <h2>Timeline</h2>
        <span>
          Step {stepCount > 0 ? currentStep + 1 : 0}/{stepCount}
        </span>
      </div>

      <div className="timeline-scrub">
        <input
          type="range"
          min={0}
          max={maxIdx}
          value={Math.min(currentStep, maxIdx)}
          onChange={(event) => setStep(Number(event.target.value))}
          disabled={!trace}
        />
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
      </div>

      <div className="step-ribbon">
        {trace?.steps.map((step) => (
          <button
            key={step.step_idx}
            onClick={() => setStep(step.step_idx)}
            className={`step-chip ${actionTone(step.action_type)} ${step.step_idx === currentStep ? "active" : ""}`}
            title={`#${step.step_idx + 1} ${step.action_name} | reward ${step.reward.toFixed(2)}`}
          >
            <motion.span
              animate={step.step_idx === currentStep ? { scale: [1, 1.06, 1] } : { scale: 1 }}
              transition={{ duration: 0.42, repeat: step.step_idx === currentStep ? Infinity : 0 }}
            >
              {step.step_idx + 1}
            </motion.span>
          </button>
        ))}
      </div>
    </footer>
  );
}
