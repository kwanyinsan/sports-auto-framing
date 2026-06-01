from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


BBox = Tuple[int, int, int, int]
Point = Tuple[int, int]
Color = Tuple[int, int, int]


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
PERSON_CLASS_ID = 0
SPORTS_BALL_CLASS_ID = 32


@dataclass
class PipelineConfig:
    input_path: Path
    output_dir: Path
    player_model: Path
    ball_model: Path
    player_imgsz: int
    ball_imgsz: int
    device: str
    half: bool
    player_conf: float
    ball_conf: float
    court_padding: float
    event_padding: float
    reels_output_size: Tuple[int, int]
    fixed_output_size: Tuple[int, int]
    full_frame_court_limit_sec: float
    save_debug: bool
    active_players: int = 4
    active_court_shrink: float = 0.08
    min_player_track_frames: int = 15
    min_player_inside_ratio: float = 0.35
    min_player_movement_ratio: float = 0.02
    identity_merge_max_gap_frames: int = 30
    identity_merge_score_threshold: float = 0.58
    identity_merge_max_distance_ratio: float = 0.20
    ball_min_conf_for_crop: float = 0.18
    ball_track_score_threshold: float = 0.55
    max_ball_missing_frames: int = 15
    crop_smoothing: float = 0.90
    max_crop_move_px: float = 35.0
    crop_deadzone_ratio: float = 0.25
    ball_crop_weight: float = 0.60
    predicted_ball_crop_weight: float = 0.30
    startup_player_lock_sec: float = 0.0
    player_keep_margin_ratio: float = 0.08
    max_ball_offset_ratio: float = 0.22
    focus_switch_angle_deg: float = 55.0
    focus_switch_min_frames: int = 8
    focus_player_search_frames: int = 12
    crop_plan_min_ball_score: float = 0.35
    crop_plan_min_segment_frames: int = 6
    crop_plan_min_segment_distance_ratio: float = 0.04
    crop_plan_smoothing_passes: int = 2
    crop_plan_smooth_window_frames: int = 15
    player_iou: float = 0.45
    ball_iou: float = 0.45
    court_resize_width: int = 960
    long_video_court_sample_count: int = 300


# Compatibility dataclasses kept so older helper modules still import cleanly.
@dataclass
class VisualizationConfig:
    court_line_color: Color = (0, 255, 0)
    court_polygon_color: Color = (255, 0, 0)
    all_tracks_color: Color = (0, 0, 255)
    valid_tracks_color: Color = (0, 255, 255)
    crop_color: Color = (255, 255, 255)
    text_color: Color = (255, 255, 255)
    line_thickness: int = 2
    polygon_alpha: float = 0.2
    font_scale: float = 0.55
    font_thickness: int = 1
    show_all_detections: bool = True


@dataclass
class CropConfig:
    portrait_aspect_ratio: Tuple[int, int] = (9, 16)
    width_ratio: float = 0.38
    smoothing_alpha: float = 0.2
    padding_ratio: float = 0.15
    auto_focus_player_counts: Tuple[int, int] = (2, 4)


@dataclass
class TrackingConfig:
    model_path: str = ""
    tracker_config: str = "bytetrack.yaml"
    confidence: float = 0.25
    iou: float = 0.45
    person_class_id: int = PERSON_CLASS_ID


@dataclass
class SamplingConfig:
    sample_count: int = 10
    first_seconds: float = 5.0
    save_previews: bool = True


@dataclass
class AppConfig:
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    preview_dir: Path = Path("debug_previews")


REQUIRED_BOUNDARY_LINES = [
    "left_outer_sideline",
    "right_outer_sideline",
    "near_baseline",
    "far_baseline",
]

OPTIONAL_LINES = [
    "near_kitchen_line",
    "far_kitchen_line",
    "center_line",
]

ANNOTATION_SEQUENCE = REQUIRED_BOUNDARY_LINES + OPTIONAL_LINES
