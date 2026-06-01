from __future__ import annotations

from pathlib import Path

import cv2

from config import BBox
from crop_utils import bbox_from_points, expand_bbox, fit_aspect_bbox_around_bbox
from video_io import VideoMetadata, create_video_writer, mux_original_audio, open_video_capture


def compute_fixed_16x9_crop(
    frames: list[dict],
    court_bbox: BBox,
    metadata: VideoMetadata,
    event_padding: float,
) -> BBox:
    points = [(court_bbox[0], court_bbox[1]), (court_bbox[2], court_bbox[3])]
    for frame in frames:
        for player in frame.get("players", []):
            x1, y1, x2, y2 = [int(value) for value in player["bbox"]]
            points.extend([(x1, y1), (x2, y2)])

    event_bbox = bbox_from_points(points, metadata.video_width, metadata.video_height)
    event_bbox = expand_bbox(event_bbox, event_padding, metadata.video_width, metadata.video_height)
    return fit_aspect_bbox_around_bbox(event_bbox, 16, 9, metadata.video_width, metadata.video_height)


def render_fixed_16x9(
    input_video: Path,
    output_video: Path,
    metadata: VideoMetadata,
    crop_box: BBox,
    output_size: tuple[int, int],
) -> None:
    silent_path = output_video.with_name(f"{output_video.stem}.silent{output_video.suffix}")
    capture = open_video_capture(input_video)
    writer = create_video_writer(silent_path, metadata.fps, output_size)
    x1, y1, x2, y2 = crop_box

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        crop = frame[y1:y2, x1:x2]
        resized = cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)
        writer.write(resized)

    capture.release()
    writer.release()
    mux_original_audio(input_video, silent_path, output_video)
