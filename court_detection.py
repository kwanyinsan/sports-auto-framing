from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from config import BBox
from crop_utils import clamp_bbox, expand_bbox
from video_io import VideoMetadata, open_video_capture, write_json


METHOD_NAME = "median_background_hough_bbox"
Line = tuple[int, int, int, int]


@dataclass
class CourtDetectionResult:
    status: str
    method: str
    video_width: int
    video_height: int
    fps: float
    frame_count: int
    duration_sec: float
    sampled_frame_count: int
    used_all_frames: bool
    resize_width: int
    court_bbox: list[int]
    court_padding: float
    confidence: float
    raw_line_count: int
    valid_line_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def sample_frames_for_court(
    video_path: Path,
    metadata: VideoMetadata,
    resize_width: int,
    full_frame_limit_sec: float,
    long_video_sample_count: int,
) -> tuple[list[np.ndarray], float, bool]:
    capture = open_video_capture(video_path)
    scale = resize_width / float(metadata.video_width)
    resized_height = max(1, int(round(metadata.video_height * scale)))
    frames: list[np.ndarray] = []
    use_all_frames = metadata.duration_sec <= full_frame_limit_sec and metadata.frame_count > 0

    if use_all_frames:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.resize(frame, (resize_width, resized_height), interpolation=cv2.INTER_AREA))
    else:
        sample_count = min(long_video_sample_count, max(1, metadata.frame_count))
        if metadata.frame_count > 0:
            indices = np.linspace(0, metadata.frame_count - 1, num=sample_count, dtype=np.int32)
            for frame_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                if ok:
                    frames.append(cv2.resize(frame, (resize_width, resized_height), interpolation=cv2.INTER_AREA))

    capture.release()
    return frames, scale, use_all_frames


def build_median_background(frames: Sequence[np.ndarray]) -> np.ndarray | None:
    if not frames:
        return None
    stack = np.stack(frames, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def detect_hough_lines(background: np.ndarray) -> list[Line]:
    gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    median_value = float(np.median(blurred))
    edges = cv2.Canny(
        blurred,
        int(max(30, 0.66 * median_value)),
        int(min(220, 1.33 * median_value + 40)),
    )

    height, width = gray.shape[:2]
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(40, int(min(width, height) * 0.08)),
        minLineLength=max(50, int(min(width, height) * 0.10)),
        maxLineGap=max(10, int(min(width, height) * 0.035)),
    )
    if raw is None:
        return []
    return [tuple(int(value) for value in line[0]) for line in raw]


def line_length(line: Line) -> float:
    x1, y1, x2, y2 = line
    return float(math.hypot(x2 - x1, y2 - y1))


def line_angle_deg(line: Line) -> float:
    x1, y1, x2, y2 = line
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


def angle_distance_deg(angle_a: float, angle_b: float) -> float:
    diff = abs(angle_a - angle_b) % 180.0
    return min(diff, 180.0 - diff)


def keep_dominant_angle_groups(lines: Sequence[Line], max_groups: int = 3) -> list[Line]:
    if len(lines) < 4:
        return list(lines)

    bin_size = 10.0
    weights = np.zeros(int(180 / bin_size), dtype=np.float32)
    for line in lines:
        weights[min(len(weights) - 1, int(line_angle_deg(line) // bin_size))] += line_length(line)

    dominant_bins = np.argsort(weights)[::-1][:max_groups]
    centers = [(float(index) + 0.5) * bin_size for index in dominant_bins if weights[index] > 0]
    kept = [line for line in lines if any(angle_distance_deg(line_angle_deg(line), center) <= 14.0 for center in centers)]
    return kept if len(kept) >= 4 else list(lines)


def filter_lines(lines: Sequence[Line], image_shape: tuple[int, int, int]) -> list[Line]:
    height, width = image_shape[:2]
    min_length = max(60, int(min(width, height) * 0.12))
    long_lines: list[Line] = []
    for line in lines:
        if line_length(line) < min_length:
            continue
        angle = line_angle_deg(line)
        if 75.0 <= angle <= 105.0:
            continue
        long_lines.append(line)

    lower_lines = [
        line
        for line in long_lines
        if ((line[1] + line[3]) / 2.0) >= height * 0.35
    ]
    candidates = lower_lines if len(lower_lines) >= 4 else long_lines
    return keep_dominant_angle_groups(candidates)


def estimate_bbox_from_lines(lines: Sequence[Line], frame_width: int, frame_height: int) -> BBox:
    if len(lines) < 4:
        return 0, 0, frame_width, frame_height

    points = []
    for x1, y1, x2, y2 in lines:
        points.extend([(x1, y1), (x2, y2)])
    xs = np.array([point[0] for point in points], dtype=np.float32)
    ys = np.array([point[1] for point in points], dtype=np.float32)
    return clamp_bbox(
        (
            int(round(float(np.percentile(xs, 4)))),
            int(round(float(np.percentile(ys, 4)))),
            int(round(float(np.percentile(xs, 96)))),
            int(round(float(np.percentile(ys, 96)))),
        ),
        frame_width,
        frame_height,
    )


def expand_to_minimum_coverage(bbox: BBox, frame_width: int, frame_height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    target_width = max(x2 - x1, int(round(frame_width * 0.90)))
    target_height = max(y2 - y1, int(round(frame_height * 0.50)))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return clamp_bbox(
        (
            int(round(cx - target_width / 2.0)),
            int(round(cy - target_height / 2.0)),
            int(round(cx + target_width / 2.0)),
            int(round(cy + target_height / 2.0)),
        ),
        frame_width,
        frame_height,
    )


def scale_bbox_to_original(bbox: BBox, scale: float, metadata: VideoMetadata) -> BBox:
    if scale <= 0:
        return 0, 0, metadata.video_width, metadata.video_height
    x1, y1, x2, y2 = bbox
    return clamp_bbox(
        (
            int(round(x1 / scale)),
            int(round(y1 / scale)),
            int(round(x2 / scale)),
            int(round(y2 / scale)),
        ),
        metadata.video_width,
        metadata.video_height,
    )


def calculate_confidence(valid_lines: Sequence[Line], bbox: BBox, frame_width: int, frame_height: int) -> float:
    line_count = len(valid_lines)
    total_length = sum(line_length(line) for line in valid_lines)
    x1, y1, x2, y2 = bbox
    area_ratio = ((x2 - x1) * (y2 - y1)) / max(1.0, float(frame_width * frame_height))
    line_score = min(1.0, line_count / 24.0)
    length_score = min(1.0, total_length / max(1.0, 3.0 * max(frame_width, frame_height)))
    area_score = 1.0 if 0.15 <= area_ratio <= 0.95 else max(0.25, min(1.0, area_ratio / 0.15))
    confidence = 0.45 * line_score + 0.35 * length_score + 0.20 * area_score
    if line_count < 4:
        confidence = min(confidence, 0.35)
    return float(max(0.0, min(1.0, confidence)))


def detect_court_bbox(
    video_path: Path,
    output_json: Path,
    metadata: VideoMetadata,
    court_padding: float,
    resize_width: int,
    full_frame_limit_sec: float,
    long_video_sample_count: int,
) -> CourtDetectionResult:
    frames, scale, used_all_frames = sample_frames_for_court(
        video_path=video_path,
        metadata=metadata,
        resize_width=resize_width,
        full_frame_limit_sec=full_frame_limit_sec,
        long_video_sample_count=long_video_sample_count,
    )
    resized_width = frames[0].shape[1] if frames else resize_width
    resized_height = frames[0].shape[0] if frames else max(1, int(round(metadata.video_height * scale)))

    background = build_median_background(frames)
    raw_lines: list[Line] = []
    valid_lines: list[Line] = []
    resized_bbox = (0, 0, resized_width, resized_height)
    if background is not None:
        raw_lines = detect_hough_lines(background)
        valid_lines = filter_lines(raw_lines, background.shape)
        resized_bbox = estimate_bbox_from_lines(valid_lines, resized_width, resized_height)
        resized_bbox = expand_bbox(resized_bbox, court_padding, resized_width, resized_height)
        resized_bbox = expand_to_minimum_coverage(resized_bbox, resized_width, resized_height)

    court_bbox = scale_bbox_to_original(resized_bbox, scale, metadata)
    confidence = calculate_confidence(valid_lines, resized_bbox, resized_width, resized_height)
    result = CourtDetectionResult(
        status="success" if confidence >= 0.6 else "low_confidence",
        method=METHOD_NAME,
        video_width=metadata.video_width,
        video_height=metadata.video_height,
        fps=metadata.fps,
        frame_count=metadata.frame_count,
        duration_sec=metadata.duration_sec,
        sampled_frame_count=len(frames),
        used_all_frames=used_all_frames,
        resize_width=resize_width,
        court_bbox=list(court_bbox),
        court_padding=court_padding,
        confidence=round(confidence, 4),
        raw_line_count=len(raw_lines),
        valid_line_count=len(valid_lines),
    )
    write_json(output_json, result.to_dict())
    return result
