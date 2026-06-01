from __future__ import annotations

from pathlib import Path

from ball_validation import candidate_static_key, select_best_ball_candidate, validate_ball_candidates
from config import BBox
from crop_utils import crop_box_from_center
from video_io import VideoMetadata, bbox_center, write_json


def _center_from_candidate(candidate: dict) -> tuple[float, float]:
    center = candidate["center"]
    return float(center[0]), float(center[1])


def _predicted_bbox(
    center: tuple[float, float],
    bbox_size: tuple[int, int],
    metadata: VideoMetadata,
) -> BBox:
    return crop_box_from_center(center, bbox_size[0], bbox_size[1], metadata.video_width, metadata.video_height)


def _update_static_counts(raw_balls: list[dict], static_counts: dict[tuple[int, int], int]) -> None:
    for detection in raw_balls:
        bbox = [int(value) for value in detection["bbox"]]
        center = bbox_center(tuple(bbox))
        key = candidate_static_key(center)
        static_counts[key] = static_counts.get(key, 0) + 1


def build_ball_track(
    raw_ball_detections: dict[int, list[dict]],
    active_player_boxes_by_frame: dict[int, list[dict]],
    court_bbox: BBox,
    metadata: VideoMetadata,
    ball_min_conf_for_crop: float,
    ball_track_score_threshold: float,
    max_missing_frames: int,
    output_json: Path | None = None,
) -> list[dict]:
    frame_count = metadata.frame_count or (max(raw_ball_detections.keys(), default=-1) + 1)
    track: list[dict] = []
    last_center: tuple[float, float] | None = None
    velocity = (0.0, 0.0)
    last_bbox_size = (10, 10)
    missing_count = max_missing_frames + 1
    static_counts: dict[tuple[int, int], int] = {}

    for frame_index in range(frame_count):
        raw_balls = raw_ball_detections.get(frame_index, [])
        _update_static_counts(raw_balls, static_counts)
        predicted_center = None
        if last_center is not None:
            predicted_center = (last_center[0] + velocity[0], last_center[1] + velocity[1])

        candidates = validate_ball_candidates(
            raw_balls=raw_balls,
            active_players=active_player_boxes_by_frame.get(frame_index, []),
            court_bbox=court_bbox,
            metadata=metadata,
            previous_center=last_center,
            predicted_center=predicted_center,
            static_counts=static_counts,
            ball_min_conf_for_crop=ball_min_conf_for_crop,
            missing_count=missing_count,
        )
        best = select_best_ball_candidate(candidates)

        if best is not None and float(best["ball_score"]) >= ball_track_score_threshold:
            center = _center_from_candidate(best)
            if last_center is not None:
                velocity = (
                    0.55 * velocity[0] + 0.45 * (center[0] - last_center[0]),
                    0.55 * velocity[1] + 0.45 * (center[1] - last_center[1]),
                )
            last_center = center
            x1, y1, x2, y2 = [int(value) for value in best["bbox"]]
            last_bbox_size = (max(4, x2 - x1), max(4, y2 - y1))
            missing_count = 0
            track.append(
                {
                    "frame": frame_index,
                    "bbox": [x1, y1, x2, y2],
                    "center": [round(center[0], 2), round(center[1], 2)],
                    "confidence": float(best["confidence"]),
                    "ball_track_score": float(best["ball_score"]),
                    "status": "detected",
                    "accepted_candidate": best,
                    "candidates": candidates[:8],
                }
            )
            continue

        missing_count += 1
        if last_center is not None and missing_count <= max_missing_frames:
            predicted_center = (
                last_center[0] + velocity[0] * missing_count,
                last_center[1] + velocity[1] * missing_count,
            )
            predicted_bbox = _predicted_bbox(predicted_center, last_bbox_size, metadata)
            track.append(
                {
                    "frame": frame_index,
                    "bbox": list(predicted_bbox),
                    "center": [round(predicted_center[0], 2), round(predicted_center[1], 2)],
                    "confidence": round(max(0.01, 0.25 * (0.80 ** missing_count)), 4),
                    "ball_track_score": round(max(0.01, ball_track_score_threshold * (0.82 ** missing_count)), 4),
                    "status": "predicted",
                    "accepted_candidate": None,
                    "candidates": candidates[:8],
                }
            )
        else:
            track.append(
                {
                    "frame": frame_index,
                    "bbox": None,
                    "center": None,
                    "confidence": 0.0,
                    "ball_track_score": 0.0,
                    "status": "missing",
                    "accepted_candidate": None,
                    "candidates": candidates[:8],
                }
            )

    if output_json is not None:
        detected_frames = sum(1 for entry in track if entry["status"] == "detected")
        predicted_frames = sum(1 for entry in track if entry["status"] == "predicted")
        write_json(
            output_json,
            {
                "method": "yolo_ball_validation_velocity_prediction",
                "court_bbox": list(court_bbox),
                "ball_min_conf_for_crop": ball_min_conf_for_crop,
                "ball_track_score_threshold": ball_track_score_threshold,
                "max_missing_frames": max_missing_frames,
                "detected_frames": detected_frames,
                "predicted_frames": predicted_frames,
                "missing_frames": len(track) - detected_frames - predicted_frames,
                "frames": track,
            },
        )
    return track
