from __future__ import annotations

from pathlib import Path

import cv2

from config import BBox
from crop_utils import crop_box_from_center, crop_size_for_output, smooth_center
from video_io import VideoMetadata, bbox_center, create_video_writer, mux_original_audio, open_video_capture


def average_player_center(players: list[dict]) -> tuple[float, float] | None:
    if not players:
        return None
    centers = [bbox_center(tuple(player["bbox"])) for player in players]
    return (
        sum(center[0] for center in centers) / len(centers),
        sum(center[1] for center in centers) / len(centers),
    )


def compute_dynamic_9x16_crops(
    frames: list[dict],
    ball_track: list[dict],
    court_bbox: BBox,
    metadata: VideoMetadata,
    output_size: tuple[int, int],
    smoothing_alpha: float,
    max_move_ratio: float,
) -> list[dict]:
    crop_width, crop_height = crop_size_for_output(
        metadata.video_width,
        metadata.video_height,
        output_size[0],
        output_size[1],
    )
    court_center = bbox_center(court_bbox)
    previous_center: tuple[float, float] | None = None
    max_step = max(metadata.video_width, metadata.video_height) * max_move_ratio
    track_by_frame = {entry["frame"]: entry for entry in ball_track}
    crop_entries: list[dict] = []

    for frame in frames:
        frame_index = int(frame["frame"])
        player_center = average_player_center(frame.get("players", []))
        ball = track_by_frame.get(frame_index, {"status": "missing"})
        ball_center = tuple(ball["center"]) if ball.get("center") else None
        status = ball.get("status", "missing")

        if status == "detected" and ball_center is not None and player_center is not None:
            target = (
                0.60 * ball_center[0] + 0.40 * player_center[0],
                0.60 * ball_center[1] + 0.40 * player_center[1],
            )
            mode = "ball-following"
        elif status == "predicted" and ball_center is not None and player_center is not None:
            target = (
                0.30 * ball_center[0] + 0.70 * player_center[0],
                0.30 * ball_center[1] + 0.70 * player_center[1],
            )
            mode = "predicted-ball"
        elif player_center is not None:
            target = player_center
            mode = "player-fallback"
        else:
            target = court_center
            mode = "court-fallback"

        smoothed = smooth_center(previous_center, target, smoothing_alpha, max_step)
        crop_box = crop_box_from_center(smoothed, crop_width, crop_height, metadata.video_width, metadata.video_height)
        previous_center = bbox_center(crop_box)
        crop_entries.append(
            {
                "frame": frame_index,
                "crop_box": list(crop_box),
                "target_center": [round(target[0], 2), round(target[1], 2)],
                "mode": mode,
            }
        )

    return crop_entries


def render_dynamic_9x16(
    input_video: Path,
    output_video: Path,
    metadata: VideoMetadata,
    crop_entries: list[dict],
    output_size: tuple[int, int],
) -> None:
    silent_path = output_video.with_name(f"{output_video.stem}.silent{output_video.suffix}")
    capture = open_video_capture(input_video)
    writer = create_video_writer(silent_path, metadata.fps, output_size)
    crops_by_frame = {
        entry["frame"]: tuple(entry.get("dynamic_9x16_crop") or entry.get("crop_box"))
        for entry in crop_entries
    }
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        x1, y1, x2, y2 = crops_by_frame.get(frame_index, (0, 0, metadata.video_width, metadata.video_height))
        crop = frame[y1:y2, x1:x2]
        resized = cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)
        writer.write(resized)
        frame_index += 1

    capture.release()
    writer.release()
    mux_original_audio(input_video, silent_path, output_video)
