#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    sync::Mutex,
    thread,
};
use tauri::{Manager, State};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FaceEvent {
    label: String,
    confidence: f32,
    camera: String,
    timestamp: DateTime<Utc>,
    source: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClusterRecord {
    id: String,
    faces: usize,
    label: String,
    status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IngestStatus {
    connected: bool,
    stream_url: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum InboundMessage {
    Event(FaceEvent),
    Clusters(Vec<ClusterRecord>),
}

struct AppState {
    stream_url: Mutex<Option<String>>,
    data_dir: Mutex<PathBuf>,
}

/// Simple health check command to keep the Tauri bridge alive.
#[tauri::command]
fn ping() -> &'static str {
    "pong"
}

/// Connect or swap the current ingest stream.
#[tauri::command]
fn connect_stream(url: String, state: State<AppState>) -> Result<(), String> {
    let mut slot = state
        .stream_url
        .lock()
        .map_err(|e| format!("state poisoned: {e}"))?;
    *slot = Some(url);
    Ok(())
}

/// Stop ingesting from the active stream.
#[tauri::command]
fn disconnect_stream(state: State<AppState>) -> Result<(), String> {
    let mut slot = state
        .stream_url
        .lock()
        .map_err(|e| format!("state poisoned: {e}"))?;
    *slot = None;
    Ok(())
}

/// Expose ingest status to the UI.
#[tauri::command]
fn ingest_status(state: State<AppState>) -> Result<IngestStatus, String> {
    let url = state
        .stream_url
        .lock()
        .map_err(|e| format!("state poisoned: {e}"))?
        .clone();
    Ok(IngestStatus {
        connected: url.is_some(),
        stream_url: url,
    })
}

/// Persist an append-only face event entry.
#[tauri::command]
fn log_face_event(event: FaceEvent, state: State<AppState>) -> Result<(), String> {
    let log_path = events_path(&data_dir(&state));
    append_json_line(&log_path, &event).map_err(|e| e.to_string())
}

/// Return the most recent face events, newest first.
#[tauri::command]
fn list_face_events(limit: Option<usize>, state: State<AppState>) -> Result<Vec<FaceEvent>, String> {
    let log_path = events_path(&data_dir(&state));
    let events = read_json_lines::<FaceEvent>(&log_path).map_err(|e| e.to_string())?;
    let mut items: Vec<FaceEvent> = events.into_iter().rev().collect();
    if let Some(max) = limit {
        items.truncate(max);
    }
    Ok(items)
}

/// Store cluster records (overwrite set).
#[tauri::command]
fn save_clusters(clusters: Vec<ClusterRecord>, state: State<AppState>) -> Result<(), String> {
    let cluster_path = clusters_path(&data_dir(&state));
    let data = serde_json::to_vec_pretty(&clusters).map_err(|e| e.to_string())?;
    fs::create_dir_all(cluster_path.parent().unwrap_or_else(|| Path::new("."))).map_err(|e| e.to_string())?;
    fs::write(cluster_path, data).map_err(|e| e.to_string())
}

/// Load cluster records.
#[tauri::command]
fn list_clusters(state: State<AppState>) -> Result<Vec<ClusterRecord>, String> {
    let cluster_path = clusters_path(&data_dir(&state));
    if !cluster_path.exists() {
        return Ok(Vec::new());
    }
    let bytes = fs::read(cluster_path).map_err(|e| e.to_string())?;
    let clusters: Vec<ClusterRecord> = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    Ok(clusters)
}

fn handle_ipc_message(msg: InboundMessage, app: &tauri::AppHandle) {
    let state: State<AppState> = app.state();
    match msg {
        InboundMessage::Event(ev) => {
            if let Err(e) = log_face_event(ev, state.clone()) {
                eprintln!("IPC event persist failed: {e}");
            }
        }
        InboundMessage::Clusters(list) => {
            if let Err(e) = save_clusters(list, state.clone()) {
                eprintln!("IPC clusters persist failed: {e}");
            }
        }
    }
}

fn main() {
    tauri::Builder::default()
        .manage(AppState {
            stream_url: Mutex::new(None),
            data_dir: Mutex::new(PathBuf::new()),
        })
        .invoke_handler(tauri::generate_handler![
            ping,
            connect_stream,
            disconnect_stream,
            ingest_status,
            log_face_event,
            list_face_events,
            save_clusters,
            list_clusters,
        ])
        .setup(|app| {
            let data_dir = std::env::var("ARC_DATA_DIR")
                .map(PathBuf::from)
                .or_else(|_| {
                    app.path_resolver()
                        .app_data_dir()
                        .or_else(|| app.path_resolver().app_dir())
                        .or_else(|| app.path_resolver().resource_dir())
                })
                .unwrap_or_else(|_| PathBuf::from("."));
            fs::create_dir_all(&data_dir).map_err(|e| anyhow::anyhow!("failed to create data dir: {e}"))?;

            {
                let state: State<AppState> = app.state();
                let mut slot = state
                    .data_dir
                    .lock()
                    .map_err(|e| anyhow::anyhow!("state poisoned: {e}"))?;
                *slot = data_dir.clone();
            }

            let ipc_host = std::env::var("ARC_IPC_HOST").unwrap_or_else(|_| "0.0.0.0".into());
            let ipc_port: u16 = std::env::var("ARC_IPC_PORT").ok().and_then(|s| s.parse().ok()).unwrap_or(8787);
            spawn_ipc_listener(ipc_host.clone(), ipc_port, app.handle());

            println!(
                "ArcCluster shell started; data dir at {} | IPC {}:{}",
                data_dir.display(), ipc_host, ipc_port
            );
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("failed to run ArcCluster app");
}

fn spawn_ipc_listener(host: String, port: u16, app: tauri::AppHandle) {
    thread::spawn(move || {
        let addr = format!("{host}:{port}");
        let listener = match TcpListener::bind(&addr) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("IPC listen failed on {addr}: {e}");
                return;
            }
        };
        for stream in listener.incoming() {
            match stream {
                Ok(s) => {
                    let app_handle = app.clone();
                    thread::spawn(move || handle_ipc_stream(s, app_handle));
                }
                Err(e) => eprintln!("IPC accept error: {e}"),
            }
        }
    });
}

fn handle_ipc_stream(stream: TcpStream, app: tauri::AppHandle) {
    let reader = BufReader::new(stream);
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("IPC read error: {e}");
                break;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<InboundMessage>(&line) {
            Ok(msg) => handle_ipc_message(msg, &app),
            Err(e) => eprintln!("IPC parse error: {e}; line={line}"),
        }
    }
}

fn data_dir(state: &State<AppState>) -> PathBuf {
    state
        .data_dir
        .lock()
        .map(|p| p.clone())
        .unwrap_or_else(|_| PathBuf::from("."))
}

fn events_path(root: &Path) -> PathBuf {
    root.join("events.jsonl")
}

fn clusters_path(root: &Path) -> PathBuf {
    root.join("clusters.json")
}

fn append_json_line<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let line = serde_json::to_string(value)?;
    writeln!(file, "{}", line)?;
    Ok(())
}

fn read_json_lines<T: for<'a> Deserialize<'a>>(path: &Path) -> anyhow::Result<Vec<T>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut items = Vec::new();
    for line in reader.lines() {
        let text = line?;
        if text.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<T>(&text) {
            Ok(item) => items.push(item),
            Err(err) => eprintln!("skip malformed line: {err}"),
        }
    }
    Ok(items)
}
