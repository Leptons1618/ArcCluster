from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    stream_url: str = "rtsp://example/stream"
    data_dir: Path = Path("./data")
    camera_name: str = "JetsonCam"
    frame_interval: int = 2  # process every Nth frame to save compute
    detection_threshold: float = 0.6
    detector: str = "yunet"  # options: yunet, haar
    detector_model: str = "./models/face_detection_yunet_2023mar.onnx"
    embedder_model: str = "./models/face_embedding.onnx"
    embedder_backend: str = "onnx"  # options: onnx, hist
    cluster_similarity: float = 0.62
    cluster_min_samples: int = 3

    @staticmethod
    def from_env() -> "PipelineConfig":
        import os

        data_dir = Path(os.getenv("ARC_DATA_DIR", "./data"))
        stream_url = os.getenv("ARC_STREAM", "rtsp://example/stream")
        camera_name = os.getenv("ARC_CAMERA", "JetsonCam")
        frame_interval = int(os.getenv("ARC_FRAME_INTERVAL", "2"))
        detection_threshold = float(os.getenv("ARC_DET_THRESH", "0.6"))
        detector = os.getenv("ARC_DETECTOR", "yunet")
        detector_model = os.getenv("ARC_DETECTOR_MODEL", "./models/face_detection_yunet_2023mar.onnx")
        embedder_model = os.getenv("ARC_EMBEDDER", "./models/face_embedding.onnx")
        embedder_backend = os.getenv("ARC_EMBEDDER_BACKEND", "onnx")
        cluster_similarity = float(os.getenv("ARC_CLUSTER_SIM", "0.62"))
        cluster_min_samples = int(os.getenv("ARC_CLUSTER_MIN", "3"))
        return PipelineConfig(
            stream_url=stream_url,
            data_dir=data_dir,
            camera_name=camera_name,
            frame_interval=frame_interval,
            detection_threshold=detection_threshold,
            detector=detector,
            detector_model=detector_model,
            embedder_model=embedder_model,
            embedder_backend=embedder_backend,
            cluster_similarity=cluster_similarity,
            cluster_min_samples=cluster_min_samples,
        )
