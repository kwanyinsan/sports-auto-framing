from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass
from os import name as os_name
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse

import cv2

from config import BBox, SUPPORTED_VIDEO_EXTENSIONS


@dataclass
class VideoMetadata:
    input_video: str
    video_width: int
    video_height: int
    fps: float
    frame_count: int
    duration_sec: float

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_path(path_value: str | Path) -> Path:
    text = str(path_value).strip().strip('"').strip("'")
    if text.lower().startswith("file://"):
        parsed = urlparse(text)
        text = unquote(parsed.path)
        if os_name == "nt" and len(text) >= 3 and text[0] == "/" and text[2] == ":":
            text = text[1:]
    return Path(text).expanduser()


def collect_input_videos(input_path: Path) -> list[Path]:
    input_path = normalize_path(input_path)
    if not input_path.exists():
        raise RuntimeError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
            raise RuntimeError(f"Input video extension is not supported: {input_path}. Supported: {supported}")
        return [input_path]

    if not input_path.is_dir():
        raise RuntimeError(f"Input path is not a video file or directory: {input_path}")

    return sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )


def iter_video_files(input_dir: Path) -> list[Path]:
    return collect_input_videos(input_dir)


def load_video_metadata(video_path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = float(frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
    capture.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Video has invalid dimensions: {video_path}")

    return VideoMetadata(
        input_video=str(video_path),
        video_width=width,
        video_height=height,
        fps=fps if fps > 0 else 30.0,
        frame_count=frame_count,
        duration_sec=duration_sec,
    )


def open_video_capture(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return capture


def create_video_writer(output_path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30.0,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")
    return writer


def _require_ffmpeg() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is required to create social-platform-compatible H.264 MP4 outputs.")
    return ffmpeg_path


def _run_ffmpeg(command: list[str], output_path: Path) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg failed while creating {output_path}: {message}")


def mux_original_audio(input_video: Path, silent_video: Path, output_video: Path) -> None:
    ffmpeg_path = _require_ffmpeg()
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(silent_video),
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-level",
        "4.2",
        "-crf",
        "23",
        "-preset",
        "medium",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output_video),
    ]
    _run_ffmpeg(command, output_video)
    silent_video.unlink(missing_ok=True)


def encode_silent_video(silent_video: Path, output_video: Path) -> None:
    ffmpeg_path = _require_ffmpeg()
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(silent_video),
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-level",
        "4.2",
        "-crf",
        "23",
        "-preset",
        "medium",
        "-an",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    _run_ffmpeg(command, output_video)
    silent_video.unlink(missing_ok=True)


def write_json(path: Path, payload: dict | list) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_in_bbox(center: tuple[float, float], bbox: BBox) -> bool:
    x, y = center
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def collect_points_from_bboxes(bboxes: Iterable[BBox]) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for x1, y1, x2, y2 in bboxes:
        points.append((x1, y1))
        points.append((x2, y2))
    return points
