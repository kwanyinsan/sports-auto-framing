from __future__ import annotations

import math
from typing import Iterable

from config import BBox
from player_tracking import bbox_area, bbox_iou
from video_io import VideoMetadata, bbox_center, center_in_bbox


def bbox_aspect(bbox: Iterable[int]) -> float:
    x1, y1, x2, y2 = [int(value) for value in bbox]
    return (x2 - x1) / max(1.0, float(y2 - y1))


def bbox_size(bbox: Iterable[int]) -> tuple[int, int]:
    x1, y1, x2, y2 = [int(value) for value in bbox]
    return max(1, x2 - x1), max(1, y2 - y1)


def candidate_static_key(center: tuple[float, float], bin_size: int = 24) -> tuple[int, int]:
    return int(round(center[0] / bin_size)), int(round(center[1] / bin_size))


def player_overlap_penalty(ball_bbox: list[int], active_players: list[dict]) -> tuple[float, float]:
    if not active_players:
        return 0.0, 0.0
    ball_area = max(1, bbox_area(ball_bbox))
    max_ball_overlap = 0.0
    max_iou = 0.0
    for player in active_players:
        player_bbox = player["bbox"]
        x1 = max(ball_bbox[0], int(player_bbox[0]))
        y1 = max(ball_bbox[1], int(player_bbox[1]))
        x2 = min(ball_bbox[2], int(player_bbox[2]))
        y2 = min(ball_bbox[3], int(player_bbox[3]))
        overlap = max(0, x2 - x1) * max(0, y2 - y1)
        max_ball_overlap = max(max_ball_overlap, overlap / float(ball_area))
        max_iou = max(max_iou, bbox_iou(ball_bbox, player_bbox))
    if max_ball_overlap >= 0.75:
        return 0.35, max_ball_overlap
    if max_ball_overlap >= 0.40:
        return 0.20, max_ball_overlap
    if max_iou >= 0.10:
        return 0.10, max_ball_overlap
    return 0.0, max_ball_overlap


def size_score(area: int, metadata: VideoMetadata) -> float:
    frame_area = max(1, metadata.video_width * metadata.video_height)
    ideal_area = max(24.0, frame_area * 0.00008)
    if area <= 0:
        return 0.0
    return float(math.exp(-abs(math.log(area / ideal_area)) / 1.35))


def motion_score(
    center: tuple[float, float],
    previous_center: tuple[float, float] | None,
    metadata: VideoMetadata,
) -> float:
    if previous_center is None:
        return 0.70
    frame_diag = math.hypot(metadata.video_width, metadata.video_height)
    distance = math.hypot(center[0] - previous_center[0], center[1] - previous_center[1])
    if distance < 1.5:
        return 0.25
    return float(math.exp(-distance / max(1.0, frame_diag * 0.16)))


def prediction_distance_score(
    center: tuple[float, float],
    predicted_center: tuple[float, float] | None,
    metadata: VideoMetadata,
) -> tuple[float, float | None]:
    if predicted_center is None:
        return 0.65, None
    frame_diag = math.hypot(metadata.video_width, metadata.video_height)
    distance = math.hypot(center[0] - predicted_center[0], center[1] - predicted_center[1])
    return float(math.exp(-distance / max(1.0, frame_diag * 0.10))), distance


def validate_ball_candidates(
    raw_balls: list[dict],
    active_players: list[dict],
    court_bbox: BBox,
    metadata: VideoMetadata,
    previous_center: tuple[float, float] | None,
    predicted_center: tuple[float, float] | None,
    static_counts: dict[tuple[int, int], int],
    ball_min_conf_for_crop: float,
    missing_count: int,
) -> list[dict]:
    frame_area = max(1, metadata.video_width * metadata.video_height)
    min_area = max(4.0, frame_area * 0.000002)
    max_area = max(180.0, frame_area * 0.0012)
    frame_diag = math.hypot(metadata.video_width, metadata.video_height)
    candidates: list[dict] = []

    for detection in raw_balls:
        bbox = [int(value) for value in detection["bbox"]]
        center = bbox_center(tuple(bbox))
        width, height = bbox_size(bbox)
        area = bbox_area(bbox)
        aspect = bbox_aspect(bbox)
        confidence = float(detection.get("confidence", 0.0))
        inside_court = center_in_bbox(center, court_bbox)
        this_size_score = size_score(area, metadata)
        this_motion_score = motion_score(center, previous_center, metadata)
        this_prediction_score, distance_to_prediction = prediction_distance_score(center, predicted_center, metadata)
        overlap_penalty, player_overlap = player_overlap_penalty(bbox, active_players)
        static_key = candidate_static_key(center)
        static_count = static_counts.get(static_key, 0)
        static_penalty = min(0.25, max(0, static_count - 8) * 0.025)

        reject_reasons: list[str] = []
        if not inside_court:
            reject_reasons.append("outside_court")
        if area < min_area:
            reject_reasons.append("too_tiny")
        if area > max_area:
            reject_reasons.append("too_large")
        if not (0.35 <= aspect <= 2.80):
            reject_reasons.append("bad_aspect_ratio")
        if previous_center is not None and missing_count <= 3:
            distance_from_previous = math.hypot(center[0] - previous_center[0], center[1] - previous_center[1])
            if distance_from_previous > frame_diag * 0.45:
                reject_reasons.append("impossible_jump")
        if confidence < ball_min_conf_for_crop and this_prediction_score < 0.75:
            reject_reasons.append("low_confidence_without_track_match")
        if static_penalty >= 0.20 and this_motion_score < 0.45:
            reject_reasons.append("static_false_object")

        ball_score = (
            0.35 * confidence
            + 0.20 * this_size_score
            + 0.20 * this_motion_score
            + 0.15 * this_prediction_score
            + 0.10 * (1.0 if inside_court else 0.0)
            - overlap_penalty
            - static_penalty
        )
        candidates.append(
            {
                "bbox": bbox,
                "center": [round(center[0], 2), round(center[1], 2)],
                "confidence": confidence,
                "area": int(area),
                "width": int(width),
                "height": int(height),
                "aspect_ratio": round(aspect, 4),
                "inside_court": inside_court,
                "distance_to_prediction": None if distance_to_prediction is None else round(distance_to_prediction, 2),
                "movement_consistency_score": round(this_motion_score, 4),
                "size_score": round(this_size_score, 4),
                "distance_to_prediction_score": round(this_prediction_score, 4),
                "player_overlap_ratio": round(player_overlap, 4),
                "player_overlap_penalty": round(overlap_penalty, 4),
                "static_count": static_count,
                "static_object_penalty": round(static_penalty, 4),
                "ball_score": round(max(0.0, min(1.0, ball_score)), 4),
                "valid": not reject_reasons,
                "reject_reasons": reject_reasons,
            }
        )

    candidates.sort(key=lambda item: float(item["ball_score"]), reverse=True)
    return candidates


def select_best_ball_candidate(candidates: list[dict]) -> dict | None:
    valid_candidates = [candidate for candidate in candidates if candidate.get("valid")]
    if not valid_candidates:
        return None
    return max(valid_candidates, key=lambda item: float(item.get("ball_score", 0.0)))
