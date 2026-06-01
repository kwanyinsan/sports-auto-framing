from __future__ import annotations

import math
from typing import Iterable

from config import BBox
from video_io import bbox_center


def clamp_bbox(bbox: BBox, frame_width: int, frame_height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(frame_width - 1, int(round(x1))))
    y1 = max(0, min(frame_height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(frame_width, int(round(x2))))
    y2 = max(y1 + 1, min(frame_height, int(round(y2))))
    return x1, y1, x2, y2


def expand_bbox(bbox: BBox, padding: float, frame_width: int, frame_height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad_x = int(round(width * padding))
    pad_y = int(round(height * padding))
    return clamp_bbox((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), frame_width, frame_height)


def bbox_from_points(points: Iterable[tuple[int, int]], frame_width: int, frame_height: int) -> BBox:
    point_list = list(points)
    if not point_list:
        return 0, 0, frame_width, frame_height
    xs = [point[0] for point in point_list]
    ys = [point[1] for point in point_list]
    return clamp_bbox((min(xs), min(ys), max(xs), max(ys)), frame_width, frame_height)


def fit_aspect_bbox_around_bbox(
    content_bbox: BBox,
    aspect_width: int,
    aspect_height: int,
    frame_width: int,
    frame_height: int,
) -> BBox:
    x1, y1, x2, y2 = content_bbox
    content_width = max(1, x2 - x1)
    content_height = max(1, y2 - y1)
    target_aspect = aspect_width / float(aspect_height)

    crop_width = content_width
    crop_height = int(round(crop_width / target_aspect))
    if crop_height < content_height:
        crop_height = content_height
        crop_width = int(round(crop_height * target_aspect))

    crop_width = min(crop_width, frame_width)
    crop_height = min(crop_height, frame_height)
    cx, cy = bbox_center(content_bbox)
    return crop_box_from_center((cx, cy), crop_width, crop_height, frame_width, frame_height)


def crop_box_from_center(
    center: tuple[float, float],
    crop_width: int,
    crop_height: int,
    frame_width: int,
    frame_height: int,
) -> BBox:
    cx, cy = center
    crop_width = min(crop_width, frame_width)
    crop_height = min(crop_height, frame_height)
    x1 = int(round(cx - crop_width / 2.0))
    y1 = int(round(cy - crop_height / 2.0))
    x2 = x1 + crop_width
    y2 = y1 + crop_height

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > frame_width:
        shift = x2 - frame_width
        x1 = max(0, x1 - shift)
        x2 = frame_width
    if y2 > frame_height:
        shift = y2 - frame_height
        y1 = max(0, y1 - shift)
        y2 = frame_height
    return x1, y1, x2, y2


def crop_size_for_output(frame_width: int, frame_height: int, output_width: int, output_height: int) -> tuple[int, int]:
    target_aspect = output_width / float(output_height)
    crop_height = frame_height
    crop_width = int(round(crop_height * target_aspect))
    if crop_width > frame_width:
        crop_width = frame_width
        crop_height = int(round(crop_width / target_aspect))
    return crop_width, crop_height


def smooth_center(
    previous: tuple[float, float] | None,
    target: tuple[float, float],
    alpha: float,
    max_step: float,
) -> tuple[float, float]:
    if previous is None:
        return target

    raw_x = (1.0 - alpha) * previous[0] + alpha * target[0]
    raw_y = (1.0 - alpha) * previous[1] + alpha * target[1]
    dx = raw_x - previous[0]
    dy = raw_y - previous[1]
    distance = math.hypot(dx, dy)
    if distance <= max_step or distance <= 1e-6:
        return raw_x, raw_y

    scale = max_step / distance
    return previous[0] + dx * scale, previous[1] + dy * scale
