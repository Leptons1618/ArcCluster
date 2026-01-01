# ArcCluster Tauri + Dioxus Console

A local-first desktop shell (Tauri) with a Dioxus web UI scaffold for monitoring the ArcCluster face agent. Front end is built to WebAssembly and served by Tauri; everything runs locally.

## Prerequisites
- Rust toolchain (stable)
- `cargo install tauri-cli --locked`
- `cargo install dioxus-cli --locked`
- Windows: enable developer mode for side-loading if you plan to build bundles

## Run in dev mode
From the workspace root:

```
cargo tauri dev
```

Tauri will:
- run `dx serve --hot-reload` on port 1420
- open the desktop window pointing at the dev server

If you only want to preview the UI in the browser:

```
cd ui
cargo install dioxus-cli --locked # first time
dx serve --addr 127.0.0.1 --port 1420 --hot-reload true --open false
```

If `dx` is not found, add Cargo to your PATH (zsh):
```
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

## Build
Production bundle (desktop installer):

```
cargo tauri build -p arccluster-app
```

## Python (Jetson) pipeline scaffold
- Code lives in `python/`: Jetson-friendly YuNet ONNX detector (preferred) and ONNX/embedder with fallback histogram mode, plus simple online clustering.
- Install deps (on the Jetson):

```
cd python
python -m venv .venv && .venv/Scripts/activate  # or source .venv/bin/activate on Linux
pip install -r requirements.txt
```

- Run against your stream and write to the shared data dir (match Tauri via `ARC_DATA_DIR`):

```
export ARC_DATA_DIR="/data/arccluster"   # pick a writable path
python pipeline.py --stream rtsp://your/stream --data-dir "$ARC_DATA_DIR"
```

The script appends events to `events.jsonl` and clusters to `clusters.json` in that directory. The Tauri app will read the same files when `ARC_DATA_DIR` is set in its environment.

Detector / embedder switches:
- `ARC_DETECTOR=yunet|haar` (default `yunet`, needs the YuNet ONNX model path via `ARC_DETECTOR_MODEL`)
- `ARC_EMBEDDER_BACKEND=onnx|hist` (default `onnx` with `ARC_EMBEDDER` pointing to your model). Set to `hist` to disable ONNX and use the lightweight histogram fallback.
- Clustering knobs: `ARC_CLUSTER_SIM` (default 0.62) and `ARC_CLUSTER_MIN` (default 3) for cosine similarity gating.

IPC (push updates):
- Tauri listens on `ARC_IPC_HOST`/`ARC_IPC_PORT` (defaults `0.0.0.0:8787`) for JSONL messages.
- Pipeline pushes events/clusters over TCP when reachable. Override host/port via env or CLI (`--ipc-host`, `--ipc-port`).
- Payloads are JSON lines like `{ "event": { ... } }` or `{ "clusters": [ ... ] }`.

## Project layout
- `Cargo.toml` — workspace root
- `ui/` — Dioxus web UI (WASM build)
- `src-tauri/` — Tauri shell and config

## Next steps
- Wire the camera ingest controls to your RTSP/HTTP handler by implementing `connect_stream`/`disconnect_stream` in the Tauri side (see `src-tauri/src/main.rs`).
- Push real face events into the `log_face_event` command and retrieve them with `list_face_events`.
- Push and read clusters through `save_clusters`/`list_clusters`.
- Data is stored locally under the app data directory (events in `events.jsonl`, clusters in `clusters.json`).
