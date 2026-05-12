import { useEffect, useMemo, useState } from "react";

import { ApplicantStoryView } from "./components/ApplicantStoryView";
import { DeskScene } from "./components/DeskScene";
import { ExplainPanel } from "./components/ExplainPanel";
import { RulesMetricsPanel } from "./components/RulesMetricsPanel";
import { TimelinePanel } from "./components/TimelinePanel";
import { TopControls } from "./components/TopControls";
import { buildApplicantStories } from "./lib/stories";
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
  const [viewMode, setViewMode] = useState<"story" | "technical">("story");

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

  const step = getCurrentStep(trace, currentStep);
  const stepCount = trace?.steps.length ?? 0;
  const canInteract = Boolean(trace && stepCount > 0);
  const stories = useMemo(() => buildApplicantStories(trace), [trace]);
  const activeStoryIndex = useMemo(() => {
    if (stories.length === 0) {
      return 0;
    }
    const idx = stories.findIndex((story) => currentStep >= story.startStep && currentStep <= story.endStep);
    return idx >= 0 ? idx : 0;
  }, [stories, currentStep]);

  const jumpToStory = (storyIndex: number) => {
    const story = stories[storyIndex];
    if (!story) {
      return;
    }
    setStep(story.decisionStep ?? story.endStep);
  };

  const stepStoryForward = () => {
    if (stories.length === 0) {
      return;
    }
    const next = Math.min(stories.length - 1, activeStoryIndex + 1);
    jumpToStory(next);
  };

  const stepStoryBackward = () => {
    if (stories.length === 0) {
      return;
    }
    const prev = Math.max(0, activeStoryIndex - 1);
    jumpToStory(prev);
  };

  useEffect(() => {
    if (!isPlaying || !trace) {
      return;
    }
    const intervalMs = Math.max(150, Math.round(780 / speed));
    const timer = window.setInterval(() => {
      if (viewMode === "technical") {
        stepForward();
        return;
      }

      if (stories.length === 0) {
        return;
      }
      if (activeStoryIndex >= stories.length - 1) {
        togglePlay();
        return;
      }
      const next = activeStoryIndex + 1;
      jumpToStory(next);
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [isPlaying, trace, speed, viewMode, stepForward, stories, activeStoryIndex, togglePlay]);

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
        viewMode={viewMode}
        onChangeViewMode={setViewMode}
        onSelectTrace={(file) => void selectTrace(file)}
        onTogglePlay={togglePlay}
        onStepBackward={viewMode === "story" ? stepStoryBackward : stepBackward}
        onStepForward={viewMode === "story" ? stepStoryForward : stepForward}
        onRestart={restart}
        onCycleSpeed={cycleSpeed}
      />

      {viewMode === "story" ? (
        <ApplicantStoryView
          stories={stories}
          activeStoryIndex={activeStoryIndex}
          trace={trace}
          onSelectStory={jumpToStory}
        />
      ) : (
        <main className="main-layout">
          <section className="scene-column">
            <DeskScene step={step} speed={speed} />
            <ExplainPanel step={step} speed={speed} />
          </section>
          <RulesMetricsPanel step={step} trace={trace} speed={speed} />
        </main>
      )}

      {viewMode === "technical" && (
        <TimelinePanel trace={trace} currentStep={currentStep} setStep={setStep} />
      )}
    </div>
  );
}
