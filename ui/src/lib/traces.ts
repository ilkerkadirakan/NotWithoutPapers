import type { ReplayTrace, TracesManifest } from "../types";

const TRACE_BASE = "/traces";

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export async function loadManifest(): Promise<TracesManifest> {
  return fetchJson<TracesManifest>(`${TRACE_BASE}/traces_manifest.json`);
}

export async function loadTrace(fileName: string): Promise<ReplayTrace> {
  return fetchJson<ReplayTrace>(`${TRACE_BASE}/${fileName}`);
}
