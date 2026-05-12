import type { TraceManifestEntry } from "../types";

interface TopControlsProps {
  manifestEntries: TraceManifestEntry[];
  selectedFile: string | null;
  canInteract: boolean;
  isPlaying: boolean;
  speed: number;
  currentStep: number;
  stepCount: number;
  traceSeed: number | null;
  loading: boolean;
  error: string | null;
  viewMode: "story" | "technical";
  onChangeViewMode: (mode: "story" | "technical") => void;
  onSelectTrace: (file: string) => void;
  onTogglePlay: () => void;
  onStepBackward: () => void;
  onStepForward: () => void;
  onRestart: () => void;
  onCycleSpeed: () => void;
}

export function TopControls(props: TopControlsProps) {
  const {
    manifestEntries,
    selectedFile,
    canInteract,
    isPlaying,
    speed,
    currentStep,
    stepCount,
    traceSeed,
    loading,
    error,
    viewMode,
    onChangeViewMode,
    onSelectTrace,
    onTogglePlay,
    onStepBackward,
    onStepForward,
    onRestart,
    onCycleSpeed
  } = props;
  const stepLabel = stepCount > 0 ? `${currentStep + 1}/${stepCount}` : "-/-";
  const statusText = loading
    ? "Loading trace..."
    : error
      ? error
      : `Seed ${traceSeed ?? "-"} | Step ${stepLabel}`;

  return (
    <header className="top-controls">
      <div className="brand">
        <p className="brand-tag">Replay Console</p>
        <h1>NotWithoutPapers</h1>
        <p className={`brand-status ${error ? "error" : ""}`}>{statusText}</p>
      </div>

      <div className="control-row">
        <label className="trace-select">
          <span>Trace</span>
          <select
            value={selectedFile ?? ""}
            onChange={(event) => onSelectTrace(event.target.value)}
            disabled={manifestEntries.length === 0}
          >
            {manifestEntries.map((entry) => (
              <option key={entry.file} value={entry.file}>
                {entry.file}
              </option>
            ))}
          </select>
        </label>

        <div className="button-group">
          <button
            className={viewMode === "story" ? "active-mode" : ""}
            onClick={() => onChangeViewMode("story")}
          >
            Story
          </button>
          <button
            className={viewMode === "technical" ? "active-mode" : ""}
            onClick={() => onChangeViewMode("technical")}
          >
            Technical
          </button>
          <button onClick={onTogglePlay} disabled={!canInteract}>
            {isPlaying ? "Pause" : "Play"}
          </button>
          <button onClick={onStepBackward} disabled={!canInteract}>
            Prev
          </button>
          <button onClick={onStepForward} disabled={!canInteract}>
            Next
          </button>
          <button onClick={onRestart} disabled={!canInteract}>
            Restart
          </button>
          <button onClick={onCycleSpeed} disabled={!canInteract}>
            {speed.toFixed(1)}x
          </button>
        </div>
      </div>
    </header>
  );
}
