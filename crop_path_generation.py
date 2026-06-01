from __future__ import annotations

import math
from pathlib import Path

from config import BBox
from crop_utils import bbox_from_points, crop_box_from_center, crop_size_for_output, expand_bbox, fit_aspect_bbox_around_bbox
from video_io import VideoMetadata, bbox_center, write_json


def average_player_center(players: list[dict]) -> tuple[float, float] | None:
    if not players:
        return None
    centers = [bbox_center(tuple(player["bbox"])) for player in players]
    return (
        sum(center[0] for center in centers) / len(centers),
        sum(center[1] for center in centers) / len(centers),
    )


def compute_fixed_16x9_crop_from_active_players(
    court_bbox: BBox,
    active_player_boxes_by_frame: dict[int, list[dict]],
    metadata: VideoMetadata,
    event_padding: float,
) -> BBox:
    points = [(court_bbox[0], court_bbox[1]), (court_bbox[2], court_bbox[3])]
    for players in active_player_boxes_by_frame.values():
        for player in players:
            x1, y1, x2, y2 = [int(value) for value in player["bbox"]]
            points.extend([(x1, y1), (x2, y2)])
    event_bbox = bbox_from_points(points, metadata.video_width, metadata.video_height)
    event_bbox = expand_bbox(event_bbox, event_padding, metadata.video_width, metadata.video_height)
    return fit_aspect_bbox_around_bbox(event_bbox, 16, 9, metadata.video_width, metadata.video_height)


def _player_center(player: dict) -> tuple[float, float]:
    return bbox_center(tuple(player["bbox"]))


def _ball_center(ball: dict) -> tuple[float, float] | None:
    center = ball.get("center")
    if center is None:
        return None
    return float(center[0]), float(center[1])


def _action_axis(court_bbox: BBox) -> str:
    x1, y1, x2, y2 = court_bbox
    return "x" if (x2 - x1) >= (y2 - y1) else "y"


def _axis_value(point: tuple[float, float], axis: str) -> float:
    return point[0] if axis == "x" else point[1]


def _nearest_player_to_point_near_frame(
    frame_index: int,
    point: tuple[float, float],
    active_player_boxes_by_frame: dict[int, list[dict]],
    frame_count: int,
    search_frames: int,
) -> dict | None:
    best: dict | None = None
    best_score = float("inf")
    radius = max(0, int(search_frames))
    for offset in range(radius + 1):
        candidate_frames = [frame_index] if offset == 0 else [frame_index - offset, frame_index + offset]
        for candidate_frame in candidate_frames:
            if candidate_frame < 0 or candidate_frame >= frame_count:
                continue
            for player in active_player_boxes_by_frame.get(candidate_frame, []):
                center = _player_center(player)
                distance = math.hypot(center[0] - point[0], center[1] - point[1])
                score = distance + offset * 18.0
                if score < best_score:
                    best_score = score
                    best = {
                        "track_id": int(player["track_id"]),
                        "identity_id": None if player.get("identity_id") is None else int(player["identity_id"]),
                        "center": center,
                        "frame": candidate_frame,
                        "distance_to_ball": distance,
                    }
    return best


def _clean_ball_points(
    ball_track: list[dict],
    action_axis: str,
    min_ball_score: float,
    max_neighbor_gap: int,
    court_bbox: BBox,
    metadata: VideoMetadata,
) -> list[dict]:
    cx1, cy1, cx2, cy2 = court_bbox
    court_w = max(1, cx2 - cx1)
    court_h = max(1, cy2 - cy1)
    margin_x = court_w * 0.04
    margin_y = court_h * 0.08
    points: list[dict] = []
    for entry in ball_track:
        center = _ball_center(entry)
        if center is None or entry.get("status") == "missing":
            continue
        score = float(entry.get("ball_track_score", 0.0))
        if score < min_ball_score:
            continue
        if entry.get("status") != "detected" and score < max(0.50, min_ball_score + 0.15):
            continue
        if not (0 <= center[0] <= metadata.video_width and 0 <= center[1] <= metadata.video_height):
            continue
        if not (cx1 - margin_x <= center[0] <= cx2 + margin_x and cy1 - margin_y <= center[1] <= cy2 + margin_y):
            continue
        points.append(
            {
                "frame": int(entry["frame"]),
                "center": center,
                "axis_value": _axis_value(center, action_axis),
                "score": score,
                "status": entry.get("status", "unknown"),
            }
        )

    if len(points) <= 2:
        return points

    cleaned: list[dict] = []
    max_gap = max(1, max_neighbor_gap)
    for index, point in enumerate(points):
        has_neighbor = False
        if index > 0 and point["frame"] - points[index - 1]["frame"] <= max_gap:
            has_neighbor = True
        if index + 1 < len(points) and points[index + 1]["frame"] - point["frame"] <= max_gap:
            has_neighbor = True
        if has_neighbor or point["score"] >= max(0.60, min_ball_score + 0.20):
            cleaned.append(point)
    return cleaned


def _build_rally_segments(
    points: list[dict],
    court_bbox: BBox,
    action_axis: str,
    min_segment_frames: int,
    min_segment_distance_ratio: float,
    min_switch_frames: int,
) -> list[dict]:
    if len(points) < 2:
        return []

    x1, y1, x2, y2 = court_bbox
    axis_length = max(1.0, float((x2 - x1) if action_axis == "x" else (y2 - y1)))
    min_distance = axis_length * max(0.0, min_segment_distance_ratio)
    min_frames = max(1, min_segment_frames)
    min_switch_gap = max(1, min_switch_frames)

    segments: list[dict] = []
    start = points[0]
    direction = 0
    extreme = points[0]
    last_switch_frame = int(start["frame"])

    for point in points[1:]:
        displacement_from_start = point["axis_value"] - start["axis_value"]
        if direction == 0 and abs(displacement_from_start) >= min_distance:
            direction = 1 if displacement_from_start > 0 else -1
            extreme = point
            continue

        if direction == 0:
            continue

        if direction > 0:
            if point["axis_value"] >= extreme["axis_value"]:
                extreme = point
            reverse_distance = extreme["axis_value"] - point["axis_value"]
        else:
            if point["axis_value"] <= extreme["axis_value"]:
                extreme = point
            reverse_distance = point["axis_value"] - extreme["axis_value"]

        segment_frames = int(extreme["frame"]) - int(start["frame"])
        far_enough_from_last = int(extreme["frame"]) - last_switch_frame >= min_switch_gap
        if reverse_distance >= min_distance and segment_frames >= min_frames and far_enough_from_last:
            segments.append(_segment_record(start, extreme, direction, min_distance))
            start = extreme
            extreme = point
            direction = -direction
            last_switch_frame = int(start["frame"])

    if direction != 0:
        segment_distance = abs(extreme["axis_value"] - start["axis_value"])
        segment_frames = int(extreme["frame"]) - int(start["frame"])
        if segment_distance >= min_distance and segment_frames >= min_frames:
            segments.append(_segment_record(start, extreme, direction, min_distance))

    return segments


def _segment_record(start: dict, end: dict, direction: int, min_distance: float) -> dict:
    distance = abs(float(end["axis_value"]) - float(start["axis_value"]))
    return {
        "start_frame": int(start["frame"]),
        "end_frame": int(end["frame"]),
        "start_center": [round(start["center"][0], 2), round(start["center"][1], 2)],
        "end_center": [round(end["center"][0], 2), round(end["center"][1], 2)],
        "direction": "positive" if direction > 0 else "negative",
        "axis_distance": round(distance, 2),
        "minimum_axis_distance": round(min_distance, 2),
    }


def _blend_target(
    ball_center: tuple[float, float],
    focus_center: tuple[float, float] | None,
    ball_weight: float,
    court_center: tuple[float, float],
) -> tuple[float, float]:
    if focus_center is None:
        return ball_center if ball_center is not None else court_center
    weight = max(0.0, min(1.0, ball_weight))
    return (
        weight * ball_center[0] + (1.0 - weight) * focus_center[0],
        weight * ball_center[1] + (1.0 - weight) * focus_center[1],
    )


def _build_crop_keyframes(
    cleaned_points: list[dict],
    segments: list[dict],
    active_player_boxes_by_frame: dict[int, list[dict]],
    metadata: VideoMetadata,
    court_center: tuple[float, float],
    ball_weight: float,
    focus_player_search_frames: int,
) -> list[dict]:
    frame_count = metadata.frame_count or 0
    keyframes: list[dict] = []
    event_points: list[tuple[str, int, tuple[float, float]]] = []

    if cleaned_points:
        first = cleaned_points[0]
        event_points.append(("start", int(first["frame"]), first["center"]))
    for segment in segments:
        event_points.append(("hit-like-segment-end", int(segment["end_frame"]), tuple(segment["end_center"])))

    seen_frames: set[int] = set()
    for reason, frame_index, ball_center in event_points:
        if frame_index in seen_frames:
            continue
        seen_frames.add(frame_index)
        nearest = _nearest_player_to_point_near_frame(
            frame_index,
            ball_center,
            active_player_boxes_by_frame,
            frame_count,
            focus_player_search_frames,
        )
        focus_center = nearest["center"] if nearest is not None else None
        target = _blend_target(ball_center, focus_center, ball_weight, court_center)
        keyframes.append(
            {
                "frame": frame_index,
                "reason": reason,
                "ball_center": [round(ball_center[0], 2), round(ball_center[1], 2)],
                "focus_player_track_id": None if nearest is None else int(nearest["track_id"]),
                "focus_player_identity_id": None if nearest is None else nearest.get("identity_id"),
                "focus_player_center": None if focus_center is None else [round(focus_center[0], 2), round(focus_center[1], 2)],
                "focus_player_frame": None if nearest is None else int(nearest["frame"]),
                "distance_to_ball": None if nearest is None else round(float(nearest["distance_to_ball"]), 2),
                "target_center": [round(target[0], 2), round(target[1], 2)],
            }
        )

    keyframes.sort(key=lambda item: int(item["frame"]))
    return keyframes


def _fallback_keyframes(
    active_player_boxes_by_frame: dict[int, list[dict]],
    court_center: tuple[float, float],
    frame_count: int,
) -> list[dict]:
    for frame_index in range(frame_count):
        center = average_player_center(active_player_boxes_by_frame.get(frame_index, []))
        if center is not None:
            return [
                {
                    "frame": frame_index,
                    "reason": "player-fallback",
                    "ball_center": None,
                "focus_player_track_id": None,
                "focus_player_identity_id": None,
                "focus_player_center": [round(center[0], 2), round(center[1], 2)],
                    "focus_player_frame": frame_index,
                    "distance_to_ball": None,
                    "target_center": [round(center[0], 2), round(center[1], 2)],
                }
            ]
    return [
        {
            "frame": 0,
            "reason": "court-fallback",
            "ball_center": None,
            "focus_player_track_id": None,
            "focus_player_identity_id": None,
            "focus_player_center": None,
            "focus_player_frame": None,
            "distance_to_ball": None,
            "target_center": [round(court_center[0], 2), round(court_center[1], 2)],
        }
    ]


def _ball_only_keyframes(cleaned_points: list[dict]) -> list[dict]:
    keyframes: list[dict] = []
    for point in cleaned_points:
        center = point["center"]
        keyframes.append(
            {
                "frame": int(point["frame"]),
                "reason": "ball-only",
                "ball_center": [round(center[0], 2), round(center[1], 2)],
                "focus_player_track_id": None,
                "focus_player_identity_id": None,
                "focus_player_center": None,
                "focus_player_frame": None,
                "distance_to_ball": None,
                "target_center": [round(center[0], 2), round(center[1], 2)],
            }
        )
    return keyframes


def _keyframe_for_frame(frame_index: int, keyframes: list[dict]) -> dict:
    if frame_index < int(keyframes[0]["frame"]):
        return keyframes[0]
    selected = keyframes[0]
    for keyframe in keyframes:
        if int(keyframe["frame"]) <= frame_index:
            selected = keyframe
        else:
            break
    return selected


def _planned_targets_from_keyframes(keyframes: list[dict], frame_count: int) -> list[tuple[float, float]]:
    targets: list[tuple[float, float]] = []
    for frame_index in range(frame_count):
        if frame_index <= int(keyframes[0]["frame"]):
            target = keyframes[0]["target_center"]
            targets.append((float(target[0]), float(target[1])))
            continue

        if frame_index >= int(keyframes[-1]["frame"]):
            target = keyframes[-1]["target_center"]
            targets.append((float(target[0]), float(target[1])))
            continue

        left = keyframes[0]
        right = keyframes[-1]
        for index in range(len(keyframes) - 1):
            if int(keyframes[index]["frame"]) <= frame_index <= int(keyframes[index + 1]["frame"]):
                left = keyframes[index]
                right = keyframes[index + 1]
                break

        left_frame = int(left["frame"])
        right_frame = int(right["frame"])
        ratio = (frame_index - left_frame) / max(1.0, float(right_frame - left_frame))
        left_target = left["target_center"]
        right_target = right["target_center"]
        targets.append(
            (
                float(left_target[0]) + ratio * (float(right_target[0]) - float(left_target[0])),
                float(left_target[1]) + ratio * (float(right_target[1]) - float(left_target[1])),
            )
        )
    return targets


def _centered_smooth_targets(
    targets: list[tuple[float, float]],
    window_frames: int,
    passes: int,
) -> list[tuple[float, float]]:
    if not targets:
        return []
    radius = max(0, int(window_frames) // 2)
    if radius <= 0:
        return list(targets)
    result = list(targets)
    for _ in range(max(0, passes)):
        smoothed = []
        for index in range(len(result)):
            start = max(0, index - radius)
            end = min(len(result), index + radius + 1)
            window = result[start:end]
            smoothed.append(
                (
                    sum(point[0] for point in window) / len(window),
                    sum(point[1] for point in window) / len(window),
                )
            )
        result = smoothed
    return result


def _limit_step(
    previous: tuple[float, float] | None,
    target: tuple[float, float],
    max_crop_move_px: float,
) -> tuple[float, float]:
    if previous is None:
        return target
    dx = target[0] - previous[0]
    dy = target[1] - previous[1]
    distance = math.hypot(dx, dy)
    max_step = max(1.0, float(max_crop_move_px))
    if distance <= max_step or distance <= 1e-6:
        return target
    scale = max_step / distance
    return previous[0] + dx * scale, previous[1] + dy * scale


def _limit_speed_bidirectional(
    targets: list[tuple[float, float]],
    max_crop_move_px: float,
    passes: int = 2,
) -> list[tuple[float, float]]:
    if not targets:
        return []
    result = list(targets)
    for _ in range(max(1, passes)):
        forward = [result[0]]
        for target in result[1:]:
            forward.append(_limit_step(forward[-1], target, max_crop_move_px))
        backward_reversed = [forward[-1]]
        for target in reversed(forward[:-1]):
            backward_reversed.append(_limit_step(backward_reversed[-1], target, max_crop_move_px))
        result = list(reversed(backward_reversed))
    return result


def generate_crop_paths(
    fixed_16x9_crop: BBox,
    active_player_boxes_by_frame: dict[int, list[dict]],
    ball_track: list[dict],
    court_bbox: BBox,
    metadata: VideoMetadata,
    reels_output_size: tuple[int, int],
    ball_track_score_threshold: float,
    crop_smoothing: float,
    max_crop_move_px: float,
    crop_deadzone_ratio: float,
    ball_crop_weight: float = 0.60,
    predicted_ball_crop_weight: float = 0.30,
    focus_switch_angle_deg: float = 55.0,
    focus_switch_min_frames: int = 8,
    focus_player_search_frames: int = 12,
    crop_plan_min_ball_score: float = 0.35,
    crop_plan_min_segment_frames: int = 6,
    crop_plan_min_segment_distance_ratio: float = 0.04,
    crop_plan_smoothing_passes: int = 2,
    crop_plan_smooth_window_frames: int = 15,
    output_json: Path | None = None,
) -> list[dict]:
    crop_width, crop_height = crop_size_for_output(
        metadata.video_width,
        metadata.video_height,
        reels_output_size[0],
        reels_output_size[1],
    )
    frame_count = metadata.frame_count or len(ball_track)
    court_center = bbox_center(court_bbox)
    action_axis = _action_axis(court_bbox)
    ball_by_frame = {int(entry["frame"]): entry for entry in ball_track}
    clean_score = min(ball_track_score_threshold, crop_plan_min_ball_score)
    cleaned_points = _clean_ball_points(
        ball_track,
        action_axis,
        clean_score,
        max_neighbor_gap=max(focus_player_search_frames, focus_switch_min_frames),
        court_bbox=court_bbox,
        metadata=metadata,
    )
    rally_segments: list[dict] = []
    keyframes = _ball_only_keyframes(cleaned_points)
    if not keyframes:
        keyframes = _fallback_keyframes(active_player_boxes_by_frame, court_center, frame_count)
    using_ball_only = bool(cleaned_points)

    planned_targets = _planned_targets_from_keyframes(keyframes, frame_count)
    smoothed_targets = _centered_smooth_targets(
        planned_targets,
        crop_plan_smooth_window_frames,
        crop_plan_smoothing_passes,
    )
    final_targets = _limit_speed_bidirectional(smoothed_targets, max_crop_move_px, passes=2)
    crop_paths: list[dict] = []

    for frame_index in range(frame_count):
        keyframe = _keyframe_for_frame(frame_index, keyframes)
        planned_target = planned_targets[frame_index]
        smoothed_target = final_targets[frame_index]
        dynamic_crop = crop_box_from_center(smoothed_target, crop_width, crop_height, metadata.video_width, metadata.video_height)
        actual_center = bbox_center(dynamic_crop)
        crop_paths.append(
            {
                "frame": frame_index,
                "fixed_16x9_crop": list(fixed_16x9_crop),
                "dynamic_9x16_crop": list(dynamic_crop),
                "crop_mode": "ball-only" if using_ball_only else keyframe["reason"],
                "target_center": [round(planned_target[0], 2), round(planned_target[1], 2)],
                "smoothed_center": [round(actual_center[0], 2), round(actual_center[1], 2)],
                "planned_center": [round(smoothed_target[0], 2), round(smoothed_target[1], 2)],
                "action_axis": action_axis,
                "focus_player_track_id": keyframe.get("focus_player_track_id"),
                "focus_player_identity_id": keyframe.get("focus_player_identity_id"),
                "focus_player_center": keyframe.get("focus_player_center"),
                "focus_player_source": "none-ball-only" if using_ball_only else "fallback",
                "focus_event_frame": int(keyframe["frame"]),
                "focus_event_reason": keyframe["reason"],
                "rally_segment_index": _segment_index_for_frame(frame_index, rally_segments),
                "ball_status": ball_by_frame.get(frame_index, {}).get("status", "missing"),
                "ball_track_score": float(ball_by_frame.get(frame_index, {}).get("ball_track_score", 0.0)),
            }
        )

    if output_json is not None:
        write_json(
            output_json,
            {
                "method": "offline_ball_only_linear_smoothed_crop_plan",
                "reels_crop_strategy": "validated_ball_path_only",
                "fixed_16x9_crop": list(fixed_16x9_crop),
                "court_bbox": list(court_bbox),
                "action_axis": action_axis,
                "ball_track_score_threshold": ball_track_score_threshold,
                "crop_plan_min_ball_score": crop_plan_min_ball_score,
                "crop_plan_min_segment_frames": crop_plan_min_segment_frames,
                "crop_plan_min_segment_distance_ratio": crop_plan_min_segment_distance_ratio,
                "crop_plan_smoothing_passes": crop_plan_smoothing_passes,
                "crop_plan_smooth_window_frames": crop_plan_smooth_window_frames,
                "crop_smoothing": crop_smoothing,
                "max_crop_move_px": max_crop_move_px,
                "crop_deadzone_ratio": crop_deadzone_ratio,
                "ball_crop_weight": ball_crop_weight,
                "predicted_ball_crop_weight": predicted_ball_crop_weight,
                "focus_switch_angle_deg": focus_switch_angle_deg,
                "focus_switch_min_frames": focus_switch_min_frames,
                "focus_player_search_frames": focus_player_search_frames,
                "cleaned_ball_points": [
                    {
                        "frame": int(point["frame"]),
                        "center": [round(point["center"][0], 2), round(point["center"][1], 2)],
                        "score": round(float(point["score"]), 4),
                        "status": point["status"],
                    }
                    for point in cleaned_points
                ],
                "rally_segments": rally_segments,
                "crop_keyframes": keyframes,
                "frames": crop_paths,
            },
        )
    return crop_paths


def _segment_index_for_frame(frame_index: int, segments: list[dict]) -> int | None:
    for index, segment in enumerate(segments):
        if int(segment["start_frame"]) <= frame_index <= int(segment["end_frame"]):
            return index
    return None
