use dioxus::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::spawn_local;

#[derive(Clone, Serialize, Deserialize)]
struct FaceEvent {
    label: String,
    confidence: f32,
    camera: String,
    timestamp: String,
    source: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
struct Cluster {
    id: String,
    faces: usize,
    label: String,
    status: String,
}

#[derive(Clone, Serialize, Deserialize, Default)]
struct IngestStatus {
    connected: bool,
    stream_url: Option<String>,
}

#[component]
fn App() -> Element {
    let face_events = use_signal(|| sample_face_events());
    let clusters = use_signal(|| sample_clusters());
    let ingest = use_signal(|| IngestStatus::default());
    let errors = use_signal(|| Vec::<String>::new());

    let stream_url = use_signal(|| "rtsp://example/stream".to_string());

    use_effect(move || {
        refresh_all(face_events, clusters, ingest, errors);
    });

    rsx! {
        div { class: "page",
            header { class: "topbar",
                div { class: "brand",
                    span { class: "badge", "Local" },
                    h1 { "ArcCluster Console" },
                },
                div { class: "actions",
                    button { class: "ghost", "Settings" },
                    button { class: "primary", "Start Pipeline" },
                },
            },

            section { class: "grid",
                div { class: "card status",
                    h2 { "Capture" },
                    p { class: "muted", "720p @ ≤30 FPS • local RTSP/HTTP" },
                    div { class: "status-row",
                        span { class: format!("dot {}", if ingest().connected { "on" } else { "off" }) }
                        span { status_label(&ingest()) }
                    },
                    div { class: "status-row",
                        span { class: "dot off" },
                        span { "Sensors idle" },
                    },
                },

                div { class: "card video",
                    h2 { "Camera" },
                    p { class: "muted", "Preview placeholder; plug your stream." },
                    div { class: "video-frame",
                        if ingest().connected {
                            format!(
                                "Connected to \"{}\"",
                                ingest()
                                    .stream_url
                                    .clone()
                                    .unwrap_or_else(|| "unknown".into())
                            )
                        } else {
                            "No stream connected"
                        }
                    },
                    div { class: "row",
                        input {
                            class: "text",
                            value: stream_url(),
                            placeholder: "rtsp://...",
                            oninput: move |ev| stream_url.set(ev.value())
                        }
                        button { class: "primary", onclick: move |_| spawn_connect(stream_url(), ingest, errors), "Connect" },
                        button { class: "ghost", onclick: move |_| spawn_disconnect(ingest, errors), "Disconnect" },
                    },
                },
            },

            section { class: "grid two",
                div { class: "card",
                    h3 { "Recent faces" },
                    div { class: "row",
                        p { class: "muted", "Append-only log of detections" },
                        button { class: "ghost", onclick: move |_| refresh_events(face_events, errors), "Refresh" },
                    },
                    ul { class: "list",
                        face_events().iter().map(|item| rsx! {
                            li { class: "list-row",
                                div { class: format!("pill {}", item_class(&item.label)), {item.label.clone()} },
                                span { class: "muted", format!("{} • {}", item.camera, item.timestamp) },
                                span { class: "confidence", format!("{}%", (item.confidence * 100.0) as i32) },
                            }
                        })
                    },
                },

                div { class: "card",
                    h3 { "Clusters" },
                    div { class: "row",
                        p { class: "muted", "Unsupervised groups ready for labeling" },
                        button { class: "ghost", onclick: move |_| refresh_clusters(clusters, errors), "Refresh" },
                    },
                    div { class: "cluster-grid",
                        clusters().iter().map(|c| rsx! {
                            div { class: "cluster",
                                div { class: "cluster-head",
                                    span { class: "cluster-id", c.id.clone() },
                                    span { class: format!("tag {}", status_class(c.status)), c.status.clone() },
                                },
                                p { class: "muted", format!("{} samples", c.faces) },
                                strong { c.label.clone() },
                                div { class: "row",
                                    button { class: "primary", "Review" },
                                    button { class: "ghost", "Merge" },
                                }
                            }
                        })
                    },
                },
            },

            section { class: "card sensors",
                h3 { "Sensor gating" },
                p { class: "muted", "Optionally pause or boost processing with external signals." },
                div { class: "chip-row",
                    span { class: "chip", "PIR: Armed" },
                    span { class: "chip", "Distance: 2.3 m" },
                    span { class: "chip", "Sound: Passive" },
                },
            },

            if !errors().is_empty() {
                div { class: "toast",
                    errors().iter().rev().take(1).map(|e| rsx!( span { {e.clone()} } )),
                },
            }
        }
    }
}

fn item_class(label: &str) -> &'static str {
    if label.eq_ignore_ascii_case("unknown") {
        "unknown"
    } else {
        "known"
    }
}

fn status_class(status: &str) -> &'static str {
    match status {
        "Known" => "ok",
        "Review" => "warn",
        _ => "off",
    }
}

fn status_label(status: &IngestStatus) -> String {
    if status.connected {
        format!("Live ingest ({})", status.stream_url.clone().unwrap_or_else(|| "stream".into()))
    } else {
        "Ingest idle".to_string()
    }
}

fn spawn_connect(url: String, mut ingest: Signal<IngestStatus>, errors: Signal<Vec<String>>) {
    spawn_invoke(async move {
        match invoke_tauri::<()> ("connect_stream", json!({ "url": url })).await {
            Ok(_) => {
                if let Ok(status) = invoke_tauri::<IngestStatus>("ingest_status", json!({})).await {
                    ingest.set(status);
                }
            }
            Err(e) => push_error(errors, format!("Connect failed: {e}")),
        }
    });
}

fn spawn_disconnect(mut ingest: Signal<IngestStatus>, errors: Signal<Vec<String>>) {
    spawn_invoke(async move {
        match invoke_tauri::<()>("disconnect_stream", json!({})).await {
            Ok(_) => ingest.set(IngestStatus { connected: false, stream_url: None }),
            Err(e) => push_error(errors, format!("Disconnect failed: {e}")),
        }
    });
}

fn sample_face_events() -> Vec<FaceEvent> {
    vec![
        FaceEvent {
            label: "Alex (Known)".into(),
            confidence: 0.92,
            camera: "Front Door".into(),
            timestamp: "14:23:10".into(),
            source: None,
        },
        FaceEvent {
            label: "Unknown".into(),
            confidence: 0.44,
            camera: "Garage".into(),
            timestamp: "14:23:02".into(),
            source: None,
        },
        FaceEvent {
            label: "Unknown".into(),
            confidence: 0.51,
            camera: "Office".into(),
            timestamp: "14:22:49".into(),
            source: None,
        },
    ]
}

fn sample_clusters() -> Vec<Cluster> {
    vec![
        Cluster {
            id: "A12".into(),
            faces: 18,
            label: "Unlabeled".into(),
            status: "Review".into(),
        },
        Cluster {
            id: "B03".into(),
            faces: 7,
            label: "Alex".into(),
            status: "Known".into(),
        },
        Cluster {
            id: "C08".into(),
            faces: 4,
            label: "Noise".into(),
            status: "Ignore".into(),
        },
    ]
}

async fn load_face_events(errors: Signal<Vec<String>>) -> Vec<FaceEvent> {
    match invoke_tauri::<Vec<FaceEvent>>("list_face_events", json!({ "limit": 25 })).await {
        Ok(items) => items,
        Err(e) => {
            push_error(errors, format!("Events load failed: {e}"));
            sample_face_events()
        }
    }
}

async fn load_clusters(errors: Signal<Vec<String>>) -> Vec<Cluster> {
    match invoke_tauri::<Vec<Cluster>>("list_clusters", json!({})).await {
        Ok(items) => items,
        Err(e) => {
            push_error(errors, format!("Clusters load failed: {e}"));
            sample_clusters()
        }
    }
}

async fn load_ingest_status(errors: Signal<Vec<String>>) -> IngestStatus {
    match invoke_tauri::<IngestStatus>("ingest_status", json!({})).await {
        Ok(status) => status,
        Err(e) => {
            push_error(errors, format!("Status failed: {e}"));
            IngestStatus::default()
        }
    }
}

#[cfg(target_arch = "wasm32")]
async fn invoke_tauri<T: for<'a> serde::de::Deserialize<'a> + 'static>(
    _cmd: &str,
    _payload: serde_json::Value,
) -> Result<T, String> {
    Err("Tauri bridge unavailable".into())
}

#[cfg(not(target_arch = "wasm32"))]
async fn invoke_tauri<T: for<'a> serde::de::Deserialize<'a> + 'static>(
    _cmd: &str,
    _payload: serde_json::Value,
) -> Result<T, String> {
    Err("Tauri bridge unavailable".into())
}

fn spawn_invoke<F>(fut: F)
where
    F: std::future::Future<Output = ()> + 'static,
{
    #[cfg(target_arch = "wasm32")]
    {
        spawn_local(fut);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = fut;
    }
}

fn push_error(mut errors: Signal<Vec<String>>, msg: String) {
    errors.with_mut(|list| {
        list.push(msg);
        if list.len() > 5 {
            let _ = list.remove(0);
        }
    });
}

fn refresh_all(
    face_events: Signal<Vec<FaceEvent>>,
    clusters: Signal<Vec<Cluster>>,
    ingest: Signal<IngestStatus>,
    errors: Signal<Vec<String>>,
) {
    refresh_events(face_events, errors);
    refresh_clusters(clusters, errors);
    refresh_ingest(ingest, errors);
}

fn refresh_events(mut face_events: Signal<Vec<FaceEvent>>, errors: Signal<Vec<String>>) {
    spawn_invoke(async move {
        let data = load_face_events(errors).await;
        face_events.set(data);
    });
}

fn refresh_clusters(mut clusters: Signal<Vec<Cluster>>, errors: Signal<Vec<String>>) {
    spawn_invoke(async move {
        let data = load_clusters(errors).await;
        clusters.set(data);
    });
}

fn refresh_ingest(mut ingest: Signal<IngestStatus>, errors: Signal<Vec<String>>) {
    spawn_invoke(async move {
        let data = load_ingest_status(errors).await;
        ingest.set(data);
    });
}

fn main() {
    dioxus::launch(App);
}
