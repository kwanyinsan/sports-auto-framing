from __future__ import annotations

import math
from pathlib import Path

from config import BBox
from crop_utils import clamp_bbox
from player_tracking import bbox_area, boxes_by_frame_for_tracks
from video_io import VideoMetadata, bbox_center, write_json


def shrink_bbox(bbox: BBox, shrink_ratio: float, frame_width: int, frame_height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    # Preserve the court's long axis so players standing near either baseline are not rejected.
    shrink_x = int(round(width * shrink_ratio)) if width < height else 0
    shrink_y = int(round(height * shrink_ratio)) if height < width else 0
    return clamp_bbox((x1 + shrink_x, y1 + shrink_y, x2 - shrink_x, y2 - shrink_y), frame_width, frame_height)


def _track_movement(frames: list[dict]) -> float:
    total = 0.0
    previous: tuple[float, float] | None = None
    for entry in frames:
        center = bbox_center(tuple(entry["bbox"]))
        if previous is not None:
            total += math.hypot(center[0] - previous[0], center[1] - previous[1])
        previous = center
    return total


def _active_court_movement(frames: list[dict]) -> float:
    total = 0.0
    previous: tuple[float, float] | None = None
    for entry in frames:
        if not entry.get("inside_active_court"):
            continue
        center = bbox_center(tuple(entry["bbox"]))
        if previous is not None:
            total += math.hypot(center[0] - previous[0], center[1] - previous[1])
        previous = center
    return total


def _average_distance_to_center(frames: list[dict], center: tuple[float, float]) -> float:
    if not frames:
        return 0.0
    total = 0.0
    for entry in frames:
        point = bbox_center(tuple(entry["bbox"]))
        total += math.hypot(point[0] - center[0], point[1] - center[1])
    return total / len(frames)


def _edge_penalty(frames: list[dict], metadata: VideoMetadata) -> float:
    if not frames:
        return 0.0
    margin_x = metadata.video_width * 0.025
    margin_y = metadata.video_height * 0.025
    edge_hits = 0
    for entry in frames:
        x1, y1, x2, y2 = [int(value) for value in entry["bbox"]]
        if x1 <= margin_x or y1 <= margin_y or x2 >= metadata.video_width - margin_x or y2 >= metadata.video_height - margin_y:
            edge_hits += 1
    edge_ratio = edge_hits / max(1, len(frames))
    return min(0.25, 0.25 * edge_ratio)


def score_player_track(track: dict, metadata: VideoMetadata, active_court_bbox: BBox) -> dict:
    frames = track.get("frames", [])
    visible_frames = int(track.get("total_visible_frames", len(frames)))
    inside_ratio = float(track.get("inside_active_court_ratio", 0.0))
    entity_key = "identity_id" if "identity_id" in track else "track_id"
    entity_id = int(track[entity_key])
    frame_diag = math.hypot(metadata.video_width, metadata.video_height)
    court_center = bbox_center(active_court_bbox)
    movement_distance = _track_movement(frames)
    active_court_movement_distance = _active_court_movement(frames)
    average_area = sum(bbox_area(entry["bbox"]) for entry in frames) / max(1, len(frames))
    average_distance = _average_distance_to_center(frames, court_center)
    movement_score = min(1.0, active_court_movement_distance / max(1.0, frame_diag * 0.30))
    duration_score = min(1.0, visible_frames / max(1.0, metadata.frame_count * 0.35))
    bbox_size_score = min(1.0, (average_area / max(1.0, metadata.video_width * metadata.video_height)) / 0.08)
    center_position_score = max(0.0, 1.0 - average_distance / max(1.0, frame_diag * 0.42))
    edge_position_penalty = _edge_penalty(frames, metadata)
    active_player_score = (
        0.55 * movement_score
        + 0.25 * inside_ratio
        + 0.10 * duration_score
        + 0.05 * bbox_size_score
        + 0.05 * center_position_score
        - edge_position_penalty
    )
    return {
        entity_key: entity_id,
        "track_ids": [int(value) for value in track.get("track_ids", [track.get("track_id", entity_id)])],
        "total_visible_frames": visible_frames,
        "inside_active_court_ratio": round(inside_ratio, 4),
        "movement_distance": round(movement_distance, 2),
        "active_court_movement_distance": round(active_court_movement_distance, 2),
        "average_bbox_area": round(average_area, 2),
        "average_distance_to_court_center": round(average_distance, 2),
        "edge_position_penalty": round(edge_position_penalty, 4),
        "normalized_active_court_movement_score": round(movement_score, 4),
        "normalized_track_duration_score": round(duration_score, 4),
        "normalized_bbox_size_score": round(bbox_size_score, 4),
        "center_position_score": round(center_position_score, 4),
        "active_player_score": round(max(0.0, active_player_score), 4),
    }


def select_active_players(
    tracks: list[dict],
    metadata: VideoMetadata,
    active_court_bbox: BBox,
    active_players: int,
    min_player_track_frames: int,
    min_player_inside_ratio: float,
    min_player_movement_ratio: float,
    output_json: Path | None = None,
) -> dict:
    min_movement_px = math.hypot(metadata.video_width, metadata.video_height) * max(0.0, min_player_movement_ratio)
    scored_tracks: list[dict] = []
    for track in tracks:
        metrics = score_player_track(track, metadata, active_court_bbox)
        rejected_reasons: list[str] = []
        if metrics["total_visible_frames"] < min_player_track_frames:
            rejected_reasons.append("too_few_frames")
        if metrics["inside_active_court_ratio"] < min_player_inside_ratio:
            rejected_reasons.append("low_inside_active_court_ratio")
        if metrics["active_court_movement_distance"] < min_movement_px:
            rejected_reasons.append("too_little_active_court_movement")
        metrics["accepted_for_ranking"] = not rejected_reasons
        metrics["rejected_reasons"] = rejected_reasons
        scored_tracks.append(metrics)

    ranked = sorted(
        [track for track in scored_tracks if track["accepted_for_ranking"]],
        key=lambda item: item["active_player_score"],
        reverse=True,
    )
    selected = ranked[: max(0, active_players)]
    uses_identities = any("identity_id" in track for track in scored_tracks)
    id_key = "identity_id" if uses_identities else "track_id"
    selected_ids = [int(track[id_key]) for track in selected]
    selected_track_ids = sorted({int(track_id) for track in selected for track_id in track.get("track_ids", [])})
    payload = {
        "method": "active_player_identity_selection" if uses_identities else "active_player_track_selection",
        "active_court_bbox": list(active_court_bbox),
        "active_players_requested": active_players,
        "min_player_track_frames": min_player_track_frames,
        "min_player_inside_ratio": min_player_inside_ratio,
        "min_player_movement_ratio": min_player_movement_ratio,
        "min_player_movement_px": round(min_movement_px, 2),
        "selected_identity_ids": selected_ids if uses_identities else [],
        "selected_track_ids": selected_track_ids if uses_identities else selected_ids,
        "selected_tracks": selected,
        "all_track_scores": scored_tracks,
    }
    if output_json is not None:
        write_json(output_json, payload)
    return payload


def active_player_boxes_by_frame(tracks: list[dict], active_track_ids: list[int]) -> dict[int, list[dict]]:
    return boxes_by_frame_for_tracks(tracks, set(active_track_ids))
