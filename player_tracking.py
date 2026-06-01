from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

from config import BBox
from video_io import VideoMetadata, bbox_center, write_json


def bbox_area(bbox: Iterable[int]) -> int:
    x1, y1, x2, y2 = [int(value) for value in bbox]
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_iou(a: Iterable[int], b: Iterable[int]) -> float:
    ax1, ay1, ax2, ay2 = [int(value) for value in a]
    bx1, by1, bx2, by2 = [int(value) for value in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = bbox_area((ax1, ay1, ax2, ay2)) + bbox_area((bx1, by1, bx2, by2)) - inter
    return inter / float(union) if union > 0 else 0.0


def foot_point(bbox: Iterable[int]) -> tuple[float, float]:
    x1, _y1, x2, y2 = [int(value) for value in bbox]
    return (x1 + x2) / 2.0, float(y2)


def point_in_bbox(point: tuple[float, float], bbox: BBox) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _match_score(
    detection_bbox: list[int],
    track_bbox: list[int],
    frame_diag: float,
) -> tuple[float, float, float]:
    iou = bbox_iou(detection_bbox, track_bbox)
    dcx, dcy = bbox_center(tuple(detection_bbox))
    tcx, tcy = bbox_center(tuple(track_bbox))
    distance = math.hypot(dcx - tcx, dcy - tcy)
    distance_score = math.exp(-distance / max(1.0, frame_diag * 0.06))
    score = 0.65 * iou + 0.35 * distance_score
    return score, iou, distance


def _detection_to_track_frame(frame_index: int, detection: dict, active_court_bbox: BBox) -> dict:
    bbox = [int(value) for value in detection["bbox"]]
    foot = foot_point(bbox)
    return {
        "frame": int(frame_index),
        "bbox": bbox,
        "confidence": float(detection.get("confidence", 0.0)),
        "foot_point": [round(foot[0], 2), round(foot[1], 2)],
        "inside_active_court": point_in_bbox(foot, active_court_bbox),
    }


def track_players(
    raw_player_detections: dict[int, list[dict]],
    metadata: VideoMetadata,
    active_court_bbox: BBox,
    output_json: Path | None = None,
    max_missing_frames: int = 8,
) -> list[dict]:
    frame_count = metadata.frame_count or (max(raw_player_detections.keys(), default=-1) + 1)
    frame_diag = math.hypot(metadata.video_width, metadata.video_height)
    open_tracks: list[dict] = []
    finished_tracks: list[dict] = []
    next_track_id = 1

    for frame_index in range(frame_count):
        detections = sorted(
            raw_player_detections.get(frame_index, []),
            key=lambda item: float(item.get("confidence", 0.0)),
            reverse=True,
        )
        candidates: list[tuple[float, int, int, float, float]] = []
        for det_index, detection in enumerate(detections):
            det_bbox = [int(value) for value in detection["bbox"]]
            for track_index, track in enumerate(open_tracks):
                if frame_index - int(track["last_frame"]) > max_missing_frames:
                    continue
                score, iou, distance = _match_score(det_bbox, track["last_bbox"], frame_diag)
                if iou >= 0.05 or distance <= frame_diag * 0.10:
                    candidates.append((score, det_index, track_index, iou, distance))

        candidates.sort(reverse=True, key=lambda item: item[0])
        used_detections: set[int] = set()
        used_tracks: set[int] = set()
        for score, det_index, track_index, _iou, _distance in candidates:
            if score < 0.16 or det_index in used_detections or track_index in used_tracks:
                continue
            detection = detections[det_index]
            entry = _detection_to_track_frame(frame_index, detection, active_court_bbox)
            track = open_tracks[track_index]
            track["frames"].append(entry)
            track["last_bbox"] = entry["bbox"]
            track["last_frame"] = frame_index
            used_detections.add(det_index)
            used_tracks.add(track_index)

        for det_index, detection in enumerate(detections):
            if det_index in used_detections:
                continue
            entry = _detection_to_track_frame(frame_index, detection, active_court_bbox)
            open_tracks.append(
                {
                    "track_id": next_track_id,
                    "first_frame": frame_index,
                    "last_frame": frame_index,
                    "last_bbox": entry["bbox"],
                    "frames": [entry],
                }
            )
            next_track_id += 1

        still_open: list[dict] = []
        for track in open_tracks:
            if frame_index - int(track["last_frame"]) > max_missing_frames:
                finished_tracks.append(track)
            else:
                still_open.append(track)
        open_tracks = still_open

    finished_tracks.extend(open_tracks)
    tracks = [_finalize_track(track) for track in finished_tracks]
    tracks.sort(key=lambda item: int(item["track_id"]))

    if output_json is not None:
        write_json(
            output_json,
            {
                "method": "greedy_iou_center_player_tracking",
                "active_court_bbox": list(active_court_bbox),
                "max_missing_frames": max_missing_frames,
                "tracks": tracks,
            },
        )
    return tracks


def _finalize_track(track: dict) -> dict:
    frames = sorted(track["frames"], key=lambda item: int(item["frame"]))
    inside_count = sum(1 for entry in frames if entry.get("inside_active_court"))
    return {
        "track_id": int(track["track_id"]),
        "first_frame": int(frames[0]["frame"]) if frames else int(track["first_frame"]),
        "last_frame": int(frames[-1]["frame"]) if frames else int(track["last_frame"]),
        "total_visible_frames": len(frames),
        "inside_active_court_frames": inside_count,
        "inside_active_court_ratio": round(inside_count / max(1, len(frames)), 4),
        "frames": frames,
    }


def boxes_by_frame_for_tracks(tracks: list[dict], track_ids: set[int] | None = None) -> dict[int, list[dict]]:
    wanted = set(track_ids) if track_ids is not None else None
    by_frame: dict[int, list[dict]] = {}
    for track in tracks:
        track_id = int(track["track_id"])
        if wanted is not None and track_id not in wanted:
            continue
        for entry in track.get("frames", []):
            frame_index = int(entry["frame"])
            item = dict(entry)
            item["track_id"] = track_id
            by_frame.setdefault(frame_index, []).append(item)
    return by_frame
