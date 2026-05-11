# UI Replay App

Animated replay player for NotWithoutPapers traces.

## Local

```bash
cmd /c npm install
cmd /c npm run dev
```

Open: `http://localhost:4173`

## Build

```bash
cmd /c npm run build
```

## Trace Source

UI expects files under `public/traces/`:

- `traces_manifest.json`
- one or more `*.json` trace files

Generate traces from project root:

```bash
python main.py trace --model-path artifacts/sweep_s42_200k_cont.zip --output-dir ui/public/traces --trace-prefix sample
```
