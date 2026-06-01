from __future__ import annotations

from pathlib import Path

import cv2

from config import BBox
from player_identity import track_to_identity_map
from player_tracking import boxes_by_frame_for_tracks
from video_io import VideoMetadata, create_video_writer, encode_silent_video, open_video_capture


def _draw_box(frame, bbox, color, label: str, thickness: int = 2) -> None:
    if bbox is None:
        return
    x1, y1, x2, y2 = [int(value) for value in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    if label:
        cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)


def _draw_center(frame, center, color, radius: int = 4) -> None:
    if center is None:
        return
    cx, cy = [int(round(float(value))) for value in center]
    cv2.circle(frame, (cx, cy), radius, color, -1, cv2.LINE_AA)


def _center_point(center) -> tuple[int, int] | None:
    if center is None:
        return None
    return int(round(float(center[0]))), int(round(float(center[1])))


def _draw_planned_crop_path(
    frame,
    crops_by_frame: dict[int, dict],
    frame_index: int,
    past_frames: int = 45,
    future_frames: int = 30,
) -> None:
    past_points = []
    for index in range(max(0, frame_index - past_frames), frame_index + 1):
        point = _center_point(crops_by_frame.get(index, {}).get("planned_center"))
        if point is not None:
            past_points.append(point)
    for start, end in zip(past_points, past_points[1:]):
        cv2.line(frame, start, end, (255, 0, 255), 2, cv2.LINE_AA)

    future_points = []
    for index in range(frame_index, frame_index + future_frames + 1):
        point = _center_point(crops_by_frame.get(index, {}).get("planned_center"))
        if point is not None:
            future_points.append(point)
    for start, end in zip(future_points, future_points[1:]):
        cv2.line(frame, start, end, (180, 0, 255), 1, cv2.LINE_AA)


def _draw_text(frame, lines: list[str]) -> None:
    y = 24
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)
        y += 23


def render_debug_video(
    input_video: Path,
    output_video: Path,
    metadata: VideoMetadata,
    raw_player_detections: dict[int, list[dict]],
    raw_ball_detections: dict[int, list[dict]],
    player_tracks: list[dict],
    player_identities: list[dict],
    active_track_ids: list[int],
    active_identity_ids: list[int],
    ball_track: list[dict],
    crop_paths: list[dict],
    court_bbox: BBox,
    active_court_bbox: BBox,
    player_model: Path,
    ball_model: Path,
    player_imgsz: int,
    ball_imgsz: int,
) -> None:
    silent_path = output_video.with_name(f"{output_video.stem}.silent{output_video.suffix}")
    capture = open_video_capture(input_video)
    writer = create_video_writer(silent_path, metadata.fps, (metadata.video_width, metadata.video_height))
    active_track_ids_set = set(active_track_ids)
    active_identity_ids_set = set(active_identity_ids)
    track_identity = track_to_identity_map(player_identities)
    tracks_by_frame = boxes_by_frame_for_tracks(player_tracks)
    ball_by_frame = {entry["frame"]: entry for entry in ball_track}
    crops_by_frame = {entry["frame"]: entry for entry in crop_paths}
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        output = frame.copy()
        crop_entry = crops_by_frame.get(frame_index, {})
        ball = ball_by_frame.get(frame_index, {"status": "missing"})
        _draw_box(output, court_bbox, (255, 255, 0), "court", 2)
        _draw_box(output, active_court_bbox, (0, 215, 255), "active court", 2)
        _draw_box(output, crop_entry.get("fixed_16x9_crop"), (255, 255, 255), "fixed 16:9", 2)
        _draw_box(output, crop_entry.get("dynamic_9x16_crop"), (255, 0, 255), "dynamic 9:16", 2)
        _draw_planned_crop_path(output, crops_by_frame, frame_index)
        _draw_center(output, crop_entry.get("planned_center"), (255, 0, 255), 4)
        focus_track_id = crop_entry.get("focus_player_track_id")
        focus_identity_id = crop_entry.get("focus_player_identity_id")

        for player in raw_player_detections.get(frame_index, []):
            _draw_box(output, player["bbox"], (128, 128, 128), f"raw person {player.get('confidence', 0.0):.2f}", 1)

        for tracked in tracks_by_frame.get(frame_index, []):
            track_id = int(tracked["track_id"])
            identity_id = track_identity.get(track_id)
            label_id = f"I{identity_id}/T{track_id}" if identity_id is not None else f"T{track_id}"
            if focus_identity_id is not None and identity_id == int(focus_identity_id):
                _draw_box(output, tracked["bbox"], (0, 255, 255), f"focus {label_id}", 4)
            elif focus_track_id is not None and track_id == int(focus_track_id):
                _draw_box(output, tracked["bbox"], (0, 255, 255), f"focus {label_id}", 4)
            elif identity_id in active_identity_ids_set or track_id in active_track_ids_set:
                color = (0, 255, 0) if track_id % 2 else (0, 255, 255)
                _draw_box(output, tracked["bbox"], color, f"active {label_id}", 3)
            else:
                _draw_box(output, tracked["bbox"], (80, 80, 220), f"ignored {label_id}", 2)

        for candidate in raw_ball_detections.get(frame_index, []):
            _draw_box(output, candidate["bbox"], (255, 128, 0), f"raw ball {candidate.get('confidence', 0.0):.2f}", 1)

        ball_status = ball.get("status", "missing")
        if ball_status == "detected":
            ball_color = (0, 255, 255)
        elif ball_status == "predicted":
            ball_color = (0, 165, 255)
        else:
            ball_color = (0, 0, 255)
        _draw_box(
            output,
            ball.get("bbox"),
            ball_color,
            f"ball {ball_status} score={float(ball.get('ball_track_score', 0.0)):.2f}",
            2,
        )
        _draw_center(output, ball.get("center"), ball_color, 5)

        _draw_text(
            output,
            [
                f"frame={frame_index}",
                f"active_identities={len(active_identity_ids_set)} raw_persons={len(raw_player_detections.get(frame_index, []))}",
                f"ball_status={ball_status} ball_score={float(ball.get('ball_track_score', 0.0)):.2f}",
                f"crop_mode={crop_entry.get('crop_mode', 'unknown')}",
                f"axis={crop_entry.get('action_axis')} segment={crop_entry.get('rally_segment_index')}",
                f"focus_identity={crop_entry.get('focus_player_identity_id')} reason={crop_entry.get('focus_event_reason')}",
                f"player={Path(player_model).name} imgsz={player_imgsz}",
                f"ball={Path(ball_model).name} imgsz={ball_imgsz}",
            ],
        )
        writer.write(output)
        frame_index += 1

    capture.release()
    writer.release()
    encode_silent_video(silent_path, output_video)
