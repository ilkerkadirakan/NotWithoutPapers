import { AnimatePresence, motion } from "framer-motion";

import { asPercent } from "../lib/presentation";
import type { ReplayTrace, RulesSnapshot, TraceStep } from "../types";

interface RulesMetricsPanelProps {
  step: TraceStep | null;
  trace: ReplayTrace | null;
  speed: number;
}

function asOnOff(value: number): string {
  return value ? "ON" : "OFF";
}

function boolClass(value: number): string {
  return value ? "yes" : "no";
}

function decisionCoverage(trace: ReplayTrace): string {
  const raw = trace.episode.stats.decision_coverage;
  if (typeof raw === "number") {
    return asPercent(raw);
  }
  return "n/a";
}

function renderRules(rules: RulesSnapshot, blink: boolean, speed: number) {
  return (
    <motion.div
      className={`rule-stack ${blink ? "update" : ""}`}
      animate={blink ? { boxShadow: ["0 0 0 rgba(0,0,0,0)", "0 0 0 2px rgba(151, 57, 19, 0.55)", "0 0 0 rgba(0,0,0,0)"] } : {}}
      transition={{ duration: Math.max(0.16, 0.28 / speed) }}
    >
      <div className="rule-row">
        <span>Permit Required</span>
        <strong className={boolClass(rules.permit_required)}>{asOnOff(rules.permit_required)}</strong>
      </div>
      <div className="rule-row">
        <span>Citizen ID Required</span>
        <strong className={boolClass(rules.id_card_required_for_citizens)}>{asOnOff(rules.id_card_required_for_citizens)}</strong>
      </div>
      <div className="rule-row">
        <span>Work Pass Required</span>
        <strong className={boolClass(rules.work_pass_required)}>{asOnOff(rules.work_pass_required)}</strong>
      </div>
      <div className="country-tags">
        {Object.entries(rules.allowed_countries_by_name).map(([country, allowed]) => (
          <span key={country} className={`country-chip ${allowed ? "allowed" : "blocked"}`}>
            {country}
          </span>
        ))}
      </div>
    </motion.div>
  );
}

export function RulesMetricsPanel({ step, trace, speed }: RulesMetricsPanelProps) {
  const rules = step?.rules_after ?? null;

  return (
    <aside className="panel side-panel">
      <section className="side-block">
        <div className="panel-header">
          <h2>Rules Board</h2>
        </div>
        {rules ? (
          <>
            {renderRules(rules, Boolean(step?.rule_update_event), speed)}
            <AnimatePresence>
              {step?.rule_update_event && (
                <motion.div
                  key={`${step.step_idx}-rule-event`}
                  className="rule-event"
                  initial={{ opacity: 0, y: -8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: Math.max(0.14, 0.24 / speed) }}
                >
                  {step.rule_update_event}
                </motion.div>
              )}
            </AnimatePresence>
          </>
        ) : (
          <div className="empty-block">Rules unavailable.</div>
        )}
      </section>

      <section className="side-block">
        <div className="panel-header">
          <h2>Episode Metrics</h2>
        </div>
        {trace ? (
          <>
            <div className="metric-inline">
              <span>Total Reward</span>
              <strong>{trace.episode.total_reward.toFixed(2)}</strong>
            </div>
            <div className="metric-inline">
              <span>Decision Coverage</span>
              <strong>{decisionCoverage(trace)}</strong>
            </div>
            <div className="metric-grid">
              {Object.entries(trace.episode.stats).map(([key, value]) => (
                <div key={key} className="metric-card">
                  <span>{key}</span>
                  <strong>{typeof value === "number" ? value.toFixed(2) : String(value)}</strong>
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="empty-block">Metrics unavailable.</div>
        )}
      </section>
    </aside>
  );
}
