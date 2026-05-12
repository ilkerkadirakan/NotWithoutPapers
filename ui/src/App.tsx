import { useEffect } from "react";

import { DeskScene } from "./components/DeskScene";
import { ExplainPanel } from "./components/ExplainPanel";
import { RulesMetricsPanel } from "./components/RulesMetricsPanel";
import { TimelinePanel } from "./components/TimelinePanel";
import { TopControls } from "./components/TopControls";
import { usePlaybackStore } from "./store/playback";
import type { ReplayTrace, TraceStep } from "./types";

function getCurrentStep(trace: ReplayTrace | null, index: number): TraceStep | null {
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
  const currentStep = usePlaybackStore((s) => s.currentStep);
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
    const intervalMs = Math.max(150, Math.round(780 / speed));
    const timer = window.setInterval(() => {
      stepForward();
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [isPlaying, speed, stepForward, trace]);

  const step = getCurrentStep(trace, currentStep);
  const stepCount = trace?.steps.length ?? 0;
  const canInteract = Boolean(trace && stepCount > 0);

  return (
    <div className="ui-shell">
      <TopControls
        manifestEntries={manifest?.traces ?? []}
        selectedFile={selectedFile}
        canInteract={canInteract}
        isPlaying={isPlaying}
        speed={speed}
        currentStep={currentStep}
        stepCount={stepCount}
        traceSeed={trace?.meta.seed ?? null}
        loading={loading}
        error={error}
        onSelectTrace={(file) => void selectTrace(file)}
        onTogglePlay={togglePlay}
        onStepBackward={stepBackward}
        onStepForward={stepForward}
        onRestart={restart}
        onCycleSpeed={cycleSpeed}
      />

      <main className="main-layout">
        <section className="scene-column">
          <DeskScene step={step} speed={speed} />
          <ExplainPanel step={step} speed={speed} />
        </section>
        <RulesMetricsPanel step={step} trace={trace} speed={speed} />
      </main>

      <TimelinePanel trace={trace} currentStep={currentStep} setStep={setStep} />
    </div>
  );
}
