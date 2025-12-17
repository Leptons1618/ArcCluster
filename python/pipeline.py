from __future__ import annotations

import argparse
import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # onnxruntime is optional; fallback logic below
    ort = None

from config import PipelineConfig
from storage import ClusterRecord, FaceEvent, append_event, load_clusters, save_clusters


@dataclass
class Detection:
    box: Tuple[int, int, int, int]
    score: float


class FaceDetector:
    def __init__(self, min_score: float = 0.6):
        self.min_score = min_score
        # Lightweight default: Haar cascade bundled with OpenCV. Replace with a faster Jetson-optimized model if desired.
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.model = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        results: List[Detection] = []
        for (x, y, w, h) in faces:
            # Haar cascades do not provide a true score; assign a placeholder confidence.
            results.append(Detection(box=(x, y, x + w, y + h), score=0.7))
        return [d for d in results if d.score >= self.min_score]


class Embedder:
    def __init__(self, model_path: str | None, backend: str = "onnx"):
        self.model_path = model_path
        self.backend = backend
        self.session = None
        if backend == "onnx" and model_path:
            if ort is None:
                raise RuntimeError("onnxruntime not available; install onnxruntime-gpu or set embedder_backend=hist")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.session = ort.InferenceSession(model_path, providers=providers)

    def embed(self, frame: np.ndarray, det: Detection) -> np.ndarray:
        x1, y1, x2, y2 = det.box
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return np.zeros(128, dtype=np.float32)
        face = cv2.resize(face, (112, 112))
        if self.backend == "onnx" and self.session is not None:
            inp = face[:, :, ::-1].astype(np.float32) / 255.0
            inp = np.transpose(inp, (2, 0, 1))[None, ...]
            outputs = self.session.run(None, {self.session.get_inputs()[0].name: inp})
            emb = outputs[0].astype(np.float32).flatten()
        else:
            # Simple color histogram as a deterministic embedding placeholder.
            hist = cv2.calcHist([face], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            emb = cv2.normalize(hist, hist).flatten().astype(np.float32)
        norm = np.linalg.norm(emb) + 1e-8
        return emb / norm


class YuNetDetector:
    def __init__(self, model_path: str, input_size: Tuple[int, int] = (320, 320), score_threshold: float = 0.6):
        self.score_threshold = score_threshold
        self.model = cv2.FaceDetectorYN.create(model=model_path, config="", input_size=input_size, score_threshold=score_threshold, nms_threshold=0.3, top_k=5000)
        self.input_size = input_size

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        self.model.setInputSize(self.input_size)
        _, faces = self.model.detect(frame)
        results: List[Detection] = []
        if faces is None:
            return results
        for face in faces:
            x, y, w_box, h_box, score = face[:5]
            if score < self.score_threshold:
                continue
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w_box), int(y + h_box)
            # clamp
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w - 1), min(y2, h - 1)
            results.append(Detection(box=(x1, y1, x2, y2), score=float(score)))
        return results


class Clusterer:
    def __init__(self, similarity_thresh: float = 0.62, min_samples: int = 3):
        self.similarity_thresh = similarity_thresh
        self.min_samples = min_samples
        self.centroids: List[np.ndarray] = []
        self.counts: List[int] = []

    def load(self, clusters: List[ClusterRecord]) -> None:
        # Without stored centroids, start fresh but preserve counts.
        self.counts = [c.faces for c in clusters]

    def update(self, embeddings: List[np.ndarray], clusters: List[ClusterRecord]) -> List[ClusterRecord]:
        for emb in embeddings:
            idx = self._match_cluster(emb)
            if idx is None:
                idx = len(self.centroids)
                self.centroids.append(emb.copy())
                self.counts.append(0)
                clusters.append(ClusterRecord(id=f"C{idx:03d}", faces=0, label="Unlabeled", status="Review"))
            # Update centroid via incremental mean.
            count = self.counts[idx] + 1
            centroid = self.centroids[idx]
            centroid = centroid + (emb - centroid) / count
            self.centroids[idx] = centroid
            self.counts[idx] = count
            clusters[idx].faces = count
        return clusters

    def _match_cluster(self, emb: np.ndarray) -> int | None:
        if not self.centroids:
            return None
        sims = [self._cosine(emb, c) for c in self.centroids]
        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= self.similarity_thresh and self.counts[best_idx] >= self.min_samples:
            return best_idx
        return None

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)


class IpcClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None

    @staticmethod
    def from_env() -> "IpcClient":
        host = os.getenv("ARC_IPC_HOST", "127.0.0.1")
        port = int(os.getenv("ARC_IPC_PORT", "8787"))
        return IpcClient(host, port)

    def _ensure(self):
        if self.sock is not None:
            return
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=1.0)
        except Exception as e:
            print(f"[IPC] connect failed: {e}")
            self.sock = None

    def send_json(self, payload: dict) -> None:
        self._ensure()
        if self.sock is None:
            return
        try:
            line = json.dumps(payload) + "\n"
            self.sock.sendall(line.encode("utf-8"))
        except Exception as e:
            print(f"[IPC] send failed: {e}")
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def send_event(self, ev: FaceEvent) -> None:
        self.send_json({"event": ev.__dict__})

    def send_clusters(self, clusters: List[ClusterRecord]) -> None:
        self.send_json({"clusters": [c.__dict__ for c in clusters]})


class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.detector = self._make_detector()
        self.embedder = Embedder(cfg.embedder_model if cfg.embedder_backend == "onnx" else None, backend=cfg.embedder_backend)
        self.clusterer = Clusterer(similarity_thresh=cfg.cluster_similarity, min_samples=cfg.cluster_min_samples)
        self.clusters: List[ClusterRecord] = load_clusters(cfg.data_dir)
        self.clusterer.load(self.clusters)
        self.ipc = IpcClient.from_env()

    def _make_detector(self):
        if self.cfg.detector == "yunet":
            return YuNetDetector(self.cfg.detector_model, score_threshold=self.cfg.detection_threshold)
        return FaceDetector(min_score=self.cfg.detection_threshold)

    def run(self, max_frames: int | None = None) -> None:
        cap = cv2.VideoCapture(self.cfg.stream_url)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open stream: {self.cfg.stream_url}")

        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("Stream ended or failed; exiting loop.")
                    break

                if frame_idx % self.cfg.frame_interval != 0:
                    frame_idx += 1
                    continue

                detections = self.detector.detect(frame)
                embeddings: List[np.ndarray] = []
                for det in detections:
                    emb = self.embedder.embed(frame, det)
                    embeddings.append(emb)
                    event = FaceEvent.new(
                        label="Unknown",
                        confidence=det.score,
                        camera=self.cfg.camera_name,
                        source=self.cfg.stream_url,
                    )
                    append_event(event, self.cfg.data_dir)
                    self.ipc.send_event(event)

                # Update clusters (placeholder logic).
                self.clusters = self.clusterer.update(embeddings, self.clusters)
                if frame_idx % 120 == 0:
                    save_clusters(self.clusters, self.cfg.data_dir)
                    self.ipc.send_clusters(self.clusters)

                frame_idx += 1
                if max_frames is not None and frame_idx >= max_frames:
                    break
        finally:
            cap.release()
            save_clusters(self.clusters, self.cfg.data_dir)
            self.ipc.send_clusters(self.clusters)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArcCluster Jetson pipeline scaffold")
    parser.add_argument("--stream", default=None, help="RTSP/HTTP stream URL")
    parser.add_argument("--data-dir", default=None, help="Directory for events/clusters")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (for testing)")
    parser.add_argument("--frame-interval", type=int, default=None, help="Process every Nth frame")
    parser.add_argument("--detector", choices=["yunet", "haar"], default=None, help="Detector backend")
    parser.add_argument("--embedder-backend", choices=["onnx", "hist"], default=None, help="Embedding backend")
    parser.add_argument("--ipc-host", default=None, help="IPC host for pushing events to Tauri (default env ARC_IPC_HOST or 127.0.0.1)")
    parser.add_argument("--ipc-port", type=int, default=None, help="IPC port (default env ARC_IPC_PORT or 8787)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig.from_env()
    if args.stream:
        cfg.stream_url = args.stream
    if args.data_dir:
        cfg.data_dir = Path(args.data_dir)
    if args.frame_interval:
        cfg.frame_interval = args.frame_interval
    if args.detector:
        cfg.detector = args.detector
    if args.embedder_backend:
        cfg.embedder_backend = args.embedder_backend
    if args.ipc_host:
        os.environ["ARC_IPC_HOST"] = args.ipc_host
    if args.ipc_port:
        os.environ["ARC_IPC_PORT"] = str(args.ipc_port)

    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(cfg)
    start = time.time()
    pipeline.run(max_frames=args.max_frames)
    elapsed = time.time() - start
    print(f"Pipeline completed in {elapsed:.2f}s; data dir: {cfg.data_dir}")


if __name__ == "__main__":
    main()
