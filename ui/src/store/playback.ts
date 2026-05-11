import { create } from "zustand";

import { loadManifest, loadTrace } from "../lib/traces";
import type { ReplayTrace, TraceManifestEntry, TracesManifest } from "../types";

const SPEED_VALUES = [0.5, 1.0, 1.5, 2.0];

interface PlaybackState {
  manifest: TracesManifest | null;
  trace: ReplayTrace | null;
  selectedFile: string | null;
  currentStep: number;
  isPlaying: boolean;
  speed: number;
  loading: boolean;
  error: string | null;

  loadManifest: () => Promise<void>;
  selectTrace: (file: string) => Promise<void>;
  setStep: (step: number) => void;
  stepForward: () => void;
  stepBackward: () => void;
  togglePlay: () => void;
  restart: () => void;
  cycleSpeed: () => void;
}

export const usePlaybackStore = create<PlaybackState>((set, get) => ({
  manifest: null,
  trace: null,
  selectedFile: null,
  currentStep: 0,
  isPlaying: false,
  speed: SPEED_VALUES[1],
  loading: false,
  error: null,

  loadManifest: async () => {
    set({ loading: true, error: null });
    try {
      const manifest = await loadManifest();
      const defaultEntry: TraceManifestEntry | undefined = manifest.traces[0];
      set({
        manifest,
        loading: false,
        selectedFile: defaultEntry?.file ?? null
      });
      if (defaultEntry) {
        await get().selectTrace(defaultEntry.file);
      }
    } catch (error) {
      set({
        loading: false,
        error: error instanceof Error ? error.message : "Unknown manifest load error"
      });
    }
  },

  selectTrace: async (file: string) => {
    set({ loading: true, error: null, selectedFile: file, isPlaying: false, currentStep: 0 });
    try {
      const trace = await loadTrace(file);
      set({ trace, loading: false });
    } catch (error) {
      set({
        loading: false,
        trace: null,
        error: error instanceof Error ? error.message : "Unknown trace load error"
      });
    }
  },

  setStep: (step: number) => {
    const trace = get().trace;
    if (!trace || trace.steps.length === 0) {
      return;
    }
    const clamped = Math.max(0, Math.min(step, trace.steps.length - 1));
    set({ currentStep: clamped });
  },

  stepForward: () => {
    const { currentStep, trace } = get();
    if (!trace || trace.steps.length === 0) {
      return;
    }
    const last = trace.steps.length - 1;
    if (currentStep >= last) {
      set({ currentStep: last, isPlaying: false });
      return;
    }
    set({ currentStep: currentStep + 1 });
  },

  stepBackward: () => {
    const { currentStep } = get();
    set({ currentStep: Math.max(0, currentStep - 1), isPlaying: false });
  },

  togglePlay: () => {
    const trace = get().trace;
    if (!trace || trace.steps.length === 0) {
      return;
    }
    set({ isPlaying: !get().isPlaying });
  },

  restart: () => {
    set({ currentStep: 0, isPlaying: false });
  },

  cycleSpeed: () => {
    const speed = get().speed;
    const idx = SPEED_VALUES.indexOf(speed);
    const next = SPEED_VALUES[(idx + 1) % SPEED_VALUES.length];
    set({ speed: next });
  }
}));
