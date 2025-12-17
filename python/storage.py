from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List


@dataclass
class FaceEvent:
    label: str
    confidence: float
    camera: str
    timestamp: str
    source: str | None = None

    @staticmethod
    def new(label: str, confidence: float, camera: str, source: str | None = None) -> "FaceEvent":
        ts = datetime.now(timezone.utc).isoformat()
        return FaceEvent(label=label, confidence=confidence, camera=camera, timestamp=ts, source=source)


@dataclass
class ClusterRecord:
    id: str
    faces: int
    label: str
    status: str


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_event(event: FaceEvent, data_dir: Path) -> None:
    events_path = data_dir / "events.jsonl"
    _ensure_dir(events_path)
    with events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(event)) + "\n")


def list_events(data_dir: Path, limit: int | None = None) -> List[FaceEvent]:
    events_path = data_dir / "events.jsonl"
    if not events_path.exists():
        return []
    items: List[FaceEvent] = []
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                items.append(FaceEvent(**data))
            except Exception:
                continue
    items.reverse()
    if limit:
        items = items[:limit]
    return items


def save_clusters(clusters: List[ClusterRecord], data_dir: Path) -> None:
    cluster_path = data_dir / "clusters.json"
    _ensure_dir(cluster_path)
    serializable = [asdict(c) for c in clusters]
    cluster_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def load_clusters(data_dir: Path) -> List[ClusterRecord]:
    cluster_path = data_dir / "clusters.json"
    if not cluster_path.exists():
        return []
    try:
        data = json.loads(cluster_path.read_text(encoding="utf-8"))
        return [ClusterRecord(**item) for item in data]
    except Exception:
        return []
