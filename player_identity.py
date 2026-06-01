from __future__ import annotations

import math
from pathlib import Path

from player_tracking import bbox_area
from video_io import VideoMetadata, bbox_center, write_json


def _track_centers(track: dict) -> list[tuple[int, tuple[float, float]]]:
    centers = []
    for entry in track.get("frames", []):
        centers.append((int(entry["frame"]), bbox_center(tuple(entry["bbox"]))))
    return centers


def _track_summary(track: dict) -> dict:
    frames = sorted(track.get("frames", []), key=lambda item: int(item["frame"]))
    centers = _track_centers({"frames": frames})
    first_center = centers[0][1] if centers else (0.0, 0.0)
    last_center = centers[-1][1] if centers else (0.0, 0.0)
    areas = [bbox_area(entry["bbox"]) for entry in frames]
    inside_count = sum(1 for entry in frames if entry.get("inside_active_court"))
    return {
        "track_id": int(track["track_id"]),
        "first_frame": int(frames[0]["frame"]) if frames else int(track.get("first_frame", 0)),
        "last_frame": int(frames[-1]["frame"]) if frames else int(track.get("last_frame", 0)),
        "first_center": first_center,
        "last_center": last_center,
        "average_area": sum(areas) / max(1, len(areas)),
        "inside_ratio": inside_count / max(1, len(frames)),
        "frame_count": len(frames),
    }


def _time_overlap(a: dict, b: dict) -> int:
    return max(0, min(int(a["last_frame"]), int(b["last_frame"])) - max(int(a["first_frame"]), int(b["first_frame"])) + 1)


def _merge_score(
    earlier: dict,
    later: dict,
    metadata: VideoMetadata,
    max_gap_frames: int,
    max_distance_ratio: float,
) -> float | None:
    gap = int(later["first_frame"]) - int(earlier["last_frame"]) - 1
    if gap < 0 or gap > max_gap_frames:
        return None

    frame_diag = math.hypot(metadata.video_width, metadata.video_height)
    distance = math.hypot(
        float(later["first_center"][0]) - float(earlier["last_center"][0]),
        float(later["first_center"][1]) - float(earlier["last_center"][1]),
    )
    max_distance = frame_diag * max_distance_ratio + gap * 10.0
    if distance > max_distance:
        return None

    time_score = 1.0 - gap / max(1.0, float(max_gap_frames))
    distance_score = 1.0 - distance / max(1.0, max_distance)
    area_score = min(earlier["average_area"], later["average_area"]) / max(1.0, max(earlier["average_area"], later["average_area"]))
    court_score = min(float(earlier["inside_ratio"]), float(later["inside_ratio"]))
    return 0.35 * time_score + 0.35 * distance_score + 0.15 * area_score + 0.15 * court_score


class _UnionFind:
    def __init__(self, values: list[int]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def _groups_overlap(group_a: list[int], group_b: list[int], summaries: dict[int, dict]) -> bool:
    for track_a in group_a:
        for track_b in group_b:
            if _time_overlap(summaries[track_a], summaries[track_b]) > 2:
                return True
    return False


def _build_identity(track_ids: list[int], tracks_by_id: dict[int, dict], identity_id: int) -> dict:
    frames: list[dict] = []
    for track_id in sorted(track_ids):
        for entry in tracks_by_id[track_id].get("frames", []):
            item = dict(entry)
            item["track_id"] = track_id
            item["identity_id"] = identity_id
            frames.append(item)
    frames.sort(key=lambda item: int(item["frame"]))
    inside_count = sum(1 for entry in frames if entry.get("inside_active_court"))
    return {
        "identity_id": identity_id,
        "track_ids": sorted(track_ids),
        "first_frame": int(frames[0]["frame"]) if frames else 0,
        "last_frame": int(frames[-1]["frame"]) if frames else 0,
        "total_visible_frames": len(frames),
        "inside_active_court_frames": inside_count,
        "inside_active_court_ratio": round(inside_count / max(1, len(frames)), 4),
        "frames": frames,
    }


def build_player_identities(
    tracks: list[dict],
    metadata: VideoMetadata,
    output_json: Path | None = None,
    max_gap_frames: int = 30,
    merge_score_threshold: float = 0.58,
    max_distance_ratio: float = 0.20,
) -> list[dict]:
    tracks_by_id = {int(track["track_id"]): track for track in tracks}
    summaries = {track_id: _track_summary(track) for track_id, track in tracks_by_id.items()}
    union_find = _UnionFind(list(tracks_by_id))
    candidates: list[tuple[float, int, int]] = []

    ordered = sorted(summaries.values(), key=lambda item: (int(item["first_frame"]), int(item["track_id"])))
    for earlier in ordered:
        for later in ordered:
            if int(later["first_frame"]) <= int(earlier["first_frame"]):
                continue
            score = _merge_score(earlier, later, metadata, max_gap_frames, max_distance_ratio)
            if score is not None and score >= merge_score_threshold:
                candidates.append((score, int(earlier["track_id"]), int(later["track_id"])))

    candidates.sort(reverse=True, key=lambda item: item[0])
    merges: list[dict] = []
    for score, track_a, track_b in candidates:
        root_a = union_find.find(track_a)
        root_b = union_find.find(track_b)
        if root_a == root_b:
            continue
        group_a = [track_id for track_id in tracks_by_id if union_find.find(track_id) == root_a]
        group_b = [track_id for track_id in tracks_by_id if union_find.find(track_id) == root_b]
        if _groups_overlap(group_a, group_b, summaries):
            continue
        union_find.union(root_a, root_b)
        merges.append({"track_a": track_a, "track_b": track_b, "merge_score": round(score, 4)})

    grouped: dict[int, list[int]] = {}
    for track_id in tracks_by_id:
        grouped.setdefault(union_find.find(track_id), []).append(track_id)

    identities = []
    for identity_id, track_ids in enumerate(sorted(grouped.values(), key=lambda ids: min(summaries[i]["first_frame"] for i in ids)), start=1):
        identities.append(_build_identity(track_ids, tracks_by_id, identity_id))

    if output_json is not None:
        write_json(
            output_json,
            {
                "method": "track_fragment_identity_merge",
                "max_gap_frames": max_gap_frames,
                "merge_score_threshold": merge_score_threshold,
                "max_distance_ratio": max_distance_ratio,
                "merge_count": len(merges),
                "merges": merges,
                "identities": identities,
            },
        )
    return identities


def boxes_by_frame_for_identities(identities: list[dict], identity_ids: set[int] | None = None) -> dict[int, list[dict]]:
    wanted = set(identity_ids) if identity_ids is not None else None
    by_frame: dict[int, list[dict]] = {}
    for identity in identities:
        identity_id = int(identity["identity_id"])
        if wanted is not None and identity_id not in wanted:
            continue
        for entry in identity.get("frames", []):
            frame_index = int(entry["frame"])
            item = dict(entry)
            item["identity_id"] = identity_id
            by_frame.setdefault(frame_index, []).append(item)
    return by_frame


def track_to_identity_map(identities: list[dict]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for identity in identities:
        identity_id = int(identity["identity_id"])
        for track_id in identity.get("track_ids", []):
            mapping[int(track_id)] = identity_id
    return mapping
