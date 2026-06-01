from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from active_player_filter import select_active_players, shrink_bbox
from ball_tracking import build_ball_track
from config import PipelineConfig
from court_detection import detect_court_bbox
from crop_path_generation import compute_fixed_16x9_crop_from_active_players, generate_crop_paths
from player_identity import boxes_by_frame_for_identities, build_player_identities
from player_tracking import track_players
from render_debug import render_debug_video
from render_dynamic_9x16 import render_dynamic_9x16
from render_fixed_16x9 import render_fixed_16x9
from timing_utils import now_seconds, timed_stage
from video_io import collect_input_videos, load_video_metadata, normalize_path, write_json
from yolo_ball_detection import run_ball_detection
from yolo_player_detection import run_player_detection


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sports auto-framing pipeline for one video or a folder of videos.")
    parser.add_argument("--input", help="Input video file or folder. Works with relative, absolute, Windows, Linux, or file:// paths.")
    parser.add_argument("--input_dir", help="Backward-compatible folder input alias.")
    parser.add_argument("--input_video", help="Single input video file.")
    parser.add_argument("--output_dir", required=True, help="Folder for generated result folders.")
    parser.add_argument("--player_model", default="models/yolo26m.pt", help="YOLO model path for person/player detection.")
    parser.add_argument("--ball_model", default="models/yolo26l.pt", help="YOLO model path for sports ball detection.")
    parser.add_argument("--player_imgsz", type=int, default=960, help="Player YOLO image size.")
    parser.add_argument("--ball_imgsz", type=int, default=960, help="Ball YOLO image size.")
    parser.add_argument("--device", default="0", help="YOLO device, for example 0 or cpu.")
    parser.add_argument("--half", type=parse_bool, default=True, help="Use half precision for YOLO inference.")
    parser.add_argument("--player_conf", type=float, default=0.25, help="Player YOLO confidence threshold.")
    parser.add_argument("--ball_conf", type=float, default=0.12, help="Ball YOLO confidence threshold.")
    parser.add_argument("--active_players", type=int, default=4, help="Maximum active player tracks to keep.")
    parser.add_argument("--active_court_shrink", type=float, default=0.08, help="Shrink only the short axis of court bbox for active-player foot-point tests.")
    parser.add_argument("--min_player_track_frames", type=int, default=15, help="Minimum visible frames for active player tracks.")
    parser.add_argument("--min_player_inside_ratio", type=float, default=0.35, help="Minimum track foot-point inside-active-court ratio.")
    parser.add_argument("--min_player_movement_ratio", type=float, default=0.02, help="Minimum movement inside active court, as a ratio of frame diagonal.")
    parser.add_argument("--identity_merge_max_gap_frames", type=int, default=30, help="Maximum frame gap for merging broken player track fragments.")
    parser.add_argument("--identity_merge_score_threshold", type=float, default=0.58, help="Minimum score for merging player track fragments into one identity.")
    parser.add_argument("--identity_merge_max_distance_ratio", type=float, default=0.20, help="Maximum merge distance as a ratio of frame diagonal.")
    parser.add_argument("--ball_min_conf_for_crop", type=float, default=0.18, help="Low-confidence ball candidates need track support below this.")
    parser.add_argument("--ball_track_score_threshold", type=float, default=0.55, help="Minimum validated ball score for detected status.")
    parser.add_argument("--max_ball_missing_frames", type=int, default=15, help="How long to predict ball position after missing detections.")
    parser.add_argument("--crop_smoothing", type=float, default=0.90, help="EMA weight for previous crop center.")
    parser.add_argument("--max_crop_move_px", type=float, default=35.0, help="Maximum dynamic crop center movement per frame.")
    parser.add_argument("--crop_deadzone_ratio", type=float, default=0.25, help="Central crop dead-zone ratio.")
    parser.add_argument("--ball_crop_weight", type=float, default=0.60, help="Ball weight in the old ball/focus-player reels target blend.")
    parser.add_argument("--predicted_ball_crop_weight", type=float, default=0.30, help="Predicted-ball weight in the old ball/focus-player reels target blend.")
    parser.add_argument("--startup_player_lock_sec", type=float, default=0.0, help="Compatibility option; the focus-player start selection is used by default.")
    parser.add_argument("--player_keep_margin_ratio", type=float, default=0.08, help="Compatibility option; ignored by the current focus-player crop.")
    parser.add_argument("--max_ball_offset_ratio", type=float, default=0.22, help="Compatibility option; ignored by the current focus-player crop.")
    parser.add_argument("--focus_switch_angle_deg", type=float, default=55.0, help="Compatibility option; current planner uses court-axis reversal instead of angle.")
    parser.add_argument("--focus_switch_min_frames", type=int, default=8, help="Minimum frames between focus-player switches.")
    parser.add_argument("--focus_player_search_frames", type=int, default=12, help="Frames around a ball event to search for the nearest active player.")
    parser.add_argument("--crop_plan_min_ball_score", type=float, default=0.35, help="Minimum ball track score used for offline reels crop planning.")
    parser.add_argument("--crop_plan_min_segment_frames", type=int, default=6, help="Minimum frames before a rally-direction segment can trigger crop movement.")
    parser.add_argument("--crop_plan_min_segment_distance_ratio", type=float, default=0.04, help="Minimum court-axis travel ratio for a valid crop-plan segment.")
    parser.add_argument("--crop_plan_smoothing_passes", type=int, default=2, help="Forward/backward smoothing passes for the offline reels crop plan.")
    parser.add_argument("--crop_plan_smooth_window_frames", type=int, default=15, help="Centered smoothing window for the offline linear reels crop path.")
    parser.add_argument("--court_padding", type=float, default=0.10, help="Padding for detected court bbox.")
    parser.add_argument("--event_padding", type=float, default=0.08, help="Padding for fixed 16:9 event bbox.")
    parser.add_argument("--reels_output_width", type=int, default=1080, help="9:16 output width.")
    parser.add_argument("--reels_output_height", type=int, default=1920, help="9:16 output height.")
    parser.add_argument("--fixed_output_width", type=int, default=1920, help="16:9 output width.")
    parser.add_argument("--fixed_output_height", type=int, default=1080, help="16:9 output height.")
    parser.add_argument("--full_frame_court_limit_sec", type=float, default=90.0, help="Use all frames for court detection under this duration.")
    parser.add_argument("--save_debug", type=parse_bool, default=True, help="Render debug_detection.mp4.")
    parser.add_argument("--player_iou", type=float, default=0.45, help="Player YOLO IoU threshold.")
    parser.add_argument("--ball_iou", type=float, default=0.45, help="Ball YOLO IoU threshold.")
    parser.add_argument("--court_resize_width", type=int, default=960, help="Court detection analysis width.")
    parser.add_argument("--long_video_court_sample_count", type=int, default=300, help="Court samples for long videos.")
    args = parser.parse_args()
    selected_inputs = [
        value
        for value in (args.input, args.input_dir, args.input_video)
        if value is not None and str(value).strip()
    ]
    if len(selected_inputs) != 1:
        parser.error("Provide exactly one input using --input, --input_dir, or --input_video.")
    return args


def selected_input_path(args: argparse.Namespace) -> Path:
    raw_path = args.input or args.input_dir or args.input_video
    return normalize_path(raw_path)


def _resolve_model_path(path: Path) -> Path:
    path = normalize_path(path)
    if path.exists():
        return path
    same_name_in_workspace = Path(path.name)
    if same_name_in_workspace.exists():
        return same_name_in_workspace
    return path


def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        input_path=selected_input_path(args),
        output_dir=normalize_path(args.output_dir),
        player_model=_resolve_model_path(Path(args.player_model)),
        ball_model=_resolve_model_path(Path(args.ball_model)),
        player_imgsz=args.player_imgsz,
        ball_imgsz=args.ball_imgsz,
        device=str(args.device),
        half=bool(args.half),
        player_conf=args.player_conf,
        ball_conf=args.ball_conf,
        court_padding=args.court_padding,
        event_padding=args.event_padding,
        reels_output_size=(args.reels_output_width, args.reels_output_height),
        fixed_output_size=(args.fixed_output_width, args.fixed_output_height),
        full_frame_court_limit_sec=args.full_frame_court_limit_sec,
        save_debug=args.save_debug,
        active_players=args.active_players,
        active_court_shrink=args.active_court_shrink,
        min_player_track_frames=args.min_player_track_frames,
        min_player_inside_ratio=args.min_player_inside_ratio,
        min_player_movement_ratio=args.min_player_movement_ratio,
        identity_merge_max_gap_frames=args.identity_merge_max_gap_frames,
        identity_merge_score_threshold=args.identity_merge_score_threshold,
        identity_merge_max_distance_ratio=args.identity_merge_max_distance_ratio,
        ball_min_conf_for_crop=args.ball_min_conf_for_crop,
        ball_track_score_threshold=args.ball_track_score_threshold,
        max_ball_missing_frames=args.max_ball_missing_frames,
        crop_smoothing=args.crop_smoothing,
        max_crop_move_px=args.max_crop_move_px,
        crop_deadzone_ratio=args.crop_deadzone_ratio,
        ball_crop_weight=args.ball_crop_weight,
        predicted_ball_crop_weight=args.predicted_ball_crop_weight,
        startup_player_lock_sec=args.startup_player_lock_sec,
        player_keep_margin_ratio=args.player_keep_margin_ratio,
        max_ball_offset_ratio=args.max_ball_offset_ratio,
        focus_switch_angle_deg=args.focus_switch_angle_deg,
        focus_switch_min_frames=args.focus_switch_min_frames,
        focus_player_search_frames=args.focus_player_search_frames,
        crop_plan_min_ball_score=args.crop_plan_min_ball_score,
        crop_plan_min_segment_frames=args.crop_plan_min_segment_frames,
        crop_plan_min_segment_distance_ratio=args.crop_plan_min_segment_distance_ratio,
        crop_plan_smoothing_passes=args.crop_plan_smoothing_passes,
        crop_plan_smooth_window_frames=args.crop_plan_smooth_window_frames,
        player_iou=args.player_iou,
        ball_iou=args.ball_iou,
        court_resize_width=args.court_resize_width,
        long_video_court_sample_count=args.long_video_court_sample_count,
    )


def logs_base(video_path: Path, output_dir: Path, config: PipelineConfig) -> dict:
    return {
        "input_video": str(video_path),
        "input_path": str(config.input_path),
        "output_folder": str(output_dir),
        "player_model": str(config.player_model),
        "ball_model": str(config.ball_model),
        "player_imgsz": config.player_imgsz,
        "ball_imgsz": config.ball_imgsz,
        "player_conf": config.player_conf,
        "ball_conf": config.ball_conf,
        "device": config.device,
        "half": config.half,
        "active_players": config.active_players,
        "active_court_shrink": config.active_court_shrink,
        "active_court_shrink_mode": "preserve_long_axis",
        "min_player_track_frames": config.min_player_track_frames,
        "min_player_inside_ratio": config.min_player_inside_ratio,
        "min_player_movement_ratio": config.min_player_movement_ratio,
        "identity_merge_max_gap_frames": config.identity_merge_max_gap_frames,
        "identity_merge_score_threshold": config.identity_merge_score_threshold,
        "identity_merge_max_distance_ratio": config.identity_merge_max_distance_ratio,
        "ball_min_conf_for_crop": config.ball_min_conf_for_crop,
        "ball_track_score_threshold": config.ball_track_score_threshold,
        "max_ball_missing_frames": config.max_ball_missing_frames,
        "crop_smoothing": config.crop_smoothing,
        "max_crop_move_px": config.max_crop_move_px,
        "crop_deadzone_ratio": config.crop_deadzone_ratio,
        "ball_crop_weight": config.ball_crop_weight,
        "predicted_ball_crop_weight": config.predicted_ball_crop_weight,
        "startup_player_lock_sec": config.startup_player_lock_sec,
        "player_keep_margin_ratio": config.player_keep_margin_ratio,
        "max_ball_offset_ratio": config.max_ball_offset_ratio,
        "focus_switch_angle_deg": config.focus_switch_angle_deg,
        "focus_switch_min_frames": config.focus_switch_min_frames,
        "focus_player_search_frames": config.focus_player_search_frames,
        "crop_plan_min_ball_score": config.crop_plan_min_ball_score,
        "crop_plan_min_segment_frames": config.crop_plan_min_segment_frames,
        "crop_plan_min_segment_distance_ratio": config.crop_plan_min_segment_distance_ratio,
        "crop_plan_smoothing_passes": config.crop_plan_smoothing_passes,
        "crop_plan_smooth_window_frames": config.crop_plan_smooth_window_frames,
        "reels_crop_method": "ball_only_validated_path",
        "output_container": "mp4",
        "output_video_codec": "h264",
        "output_audio_codec": "aac",
        "output_pixel_format": "yuv420p",
        "court_padding": config.court_padding,
        "event_padding": config.event_padding,
        "status": "running",
        "timings": {},
        "errors": [],
    }


def _frame_count(metadata, raw_players: dict[int, list[dict]], raw_balls: dict[int, list[dict]]) -> int:
    return metadata.frame_count or (max(list(raw_players.keys()) + list(raw_balls.keys()) + [-1]) + 1)


def save_raw_detections(
    output_json: Path,
    metadata,
    court_bbox,
    active_court_bbox,
    raw_players: dict[int, list[dict]],
    raw_balls: dict[int, list[dict]],
    config: PipelineConfig,
) -> None:
    frames = []
    for frame_index in range(_frame_count(metadata, raw_players, raw_balls)):
        frames.append(
            {
                "frame": frame_index,
                "raw_players": raw_players.get(frame_index, []),
                "raw_balls": raw_balls.get(frame_index, []),
            }
        )
    write_json(
        output_json,
        {
            "video": metadata.to_dict(),
            "court_bbox": list(court_bbox),
            "active_court_bbox": list(active_court_bbox),
            "player_model": str(config.player_model),
            "ball_model": str(config.ball_model),
            "player_imgsz": config.player_imgsz,
            "ball_imgsz": config.ball_imgsz,
            "player_conf": config.player_conf,
            "ball_conf": config.ball_conf,
            "frames": frames,
        },
    )


def process_one_video(video_path: Path, config: PipelineConfig) -> None:
    output_folder = config.output_dir / video_path.stem
    output_folder.mkdir(parents=True, exist_ok=True)
    log_path = output_folder / "logs.json"
    logs = logs_base(video_path, output_folder, config)
    total_start = now_seconds()

    print(f"\nProcessing video: {video_path.name}")
    try:
        metadata = load_video_metadata(video_path)
        logs["video_metadata"] = metadata.to_dict()
        timings: dict[str, float] = logs["timings"]

        with timed_stage("Court detection time", timings):
            court_result = detect_court_bbox(
                video_path=video_path,
                output_json=output_folder / "court_result.json",
                metadata=metadata,
                court_padding=config.court_padding,
                resize_width=config.court_resize_width,
                full_frame_limit_sec=config.full_frame_court_limit_sec,
                long_video_sample_count=config.long_video_court_sample_count,
            )
            court_bbox = tuple(int(value) for value in court_result.court_bbox)
            active_court_bbox = shrink_bbox(
                court_bbox,
                config.active_court_shrink,
                metadata.video_width,
                metadata.video_height,
            )
            court_payload = court_result.to_dict()
            court_payload["active_court_bbox"] = list(active_court_bbox)
            court_payload["active_court_shrink"] = config.active_court_shrink
            court_payload["active_court_shrink_mode"] = "preserve_long_axis"
            write_json(output_folder / "court_result.json", court_payload)
        logs["court_bbox"] = list(court_bbox)
        logs["active_court_bbox"] = list(active_court_bbox)

        with timed_stage("Player YOLO detection time", timings):
            raw_players = run_player_detection(
                video_path=video_path,
                output_json=None,
                model_path=config.player_model,
                imgsz=config.player_imgsz,
                conf=config.player_conf,
                iou=config.player_iou,
                device=config.device,
                half=config.half,
            )

        with timed_stage("Ball YOLO detection time", timings):
            raw_balls = run_ball_detection(
                video_path=video_path,
                output_json=None,
                model_path=config.ball_model,
                imgsz=config.ball_imgsz,
                conf=config.ball_conf,
                iou=config.ball_iou,
                device=config.device,
                half=config.half,
            )

        save_raw_detections(
            output_folder / "raw_detections.json",
            metadata,
            court_bbox,
            active_court_bbox,
            raw_players,
            raw_balls,
            config,
        )

        with timed_stage("Player track filtering time", timings):
            player_tracks = track_players(
                raw_player_detections=raw_players,
                metadata=metadata,
                active_court_bbox=active_court_bbox,
                output_json=output_folder / "player_tracks.json",
            )
            player_identities = build_player_identities(
                tracks=player_tracks,
                metadata=metadata,
                output_json=output_folder / "player_identities.json",
                max_gap_frames=config.identity_merge_max_gap_frames,
                merge_score_threshold=config.identity_merge_score_threshold,
                max_distance_ratio=config.identity_merge_max_distance_ratio,
            )
            active_payload = select_active_players(
                tracks=player_identities,
                metadata=metadata,
                active_court_bbox=active_court_bbox,
                active_players=config.active_players,
                min_player_track_frames=config.min_player_track_frames,
                min_player_inside_ratio=config.min_player_inside_ratio,
                min_player_movement_ratio=config.min_player_movement_ratio,
                output_json=output_folder / "active_players.json",
            )
            active_identity_ids = [int(value) for value in active_payload.get("selected_identity_ids", [])]
            active_track_ids = [int(value) for value in active_payload["selected_track_ids"]]
            active_boxes_by_frame = boxes_by_frame_for_identities(player_identities, set(active_identity_ids))

        with timed_stage("Ball validation/tracking time", timings):
            ball_track = build_ball_track(
                raw_ball_detections=raw_balls,
                active_player_boxes_by_frame=active_boxes_by_frame,
                court_bbox=court_bbox,
                metadata=metadata,
                ball_min_conf_for_crop=config.ball_min_conf_for_crop,
                ball_track_score_threshold=config.ball_track_score_threshold,
                max_missing_frames=config.max_ball_missing_frames,
                output_json=output_folder / "ball_track.json",
            )
            fixed_crop = compute_fixed_16x9_crop_from_active_players(
                court_bbox=court_bbox,
                active_player_boxes_by_frame=active_boxes_by_frame,
                metadata=metadata,
                event_padding=config.event_padding,
            )
            crop_paths = generate_crop_paths(
                fixed_16x9_crop=fixed_crop,
                active_player_boxes_by_frame=active_boxes_by_frame,
                ball_track=ball_track,
                court_bbox=court_bbox,
                metadata=metadata,
                reels_output_size=config.reels_output_size,
                ball_track_score_threshold=config.ball_track_score_threshold,
                crop_smoothing=config.crop_smoothing,
                max_crop_move_px=config.max_crop_move_px,
                crop_deadzone_ratio=config.crop_deadzone_ratio,
                ball_crop_weight=config.ball_crop_weight,
                predicted_ball_crop_weight=config.predicted_ball_crop_weight,
                focus_switch_angle_deg=config.focus_switch_angle_deg,
                focus_switch_min_frames=config.focus_switch_min_frames,
                focus_player_search_frames=config.focus_player_search_frames,
                crop_plan_min_ball_score=config.crop_plan_min_ball_score,
                crop_plan_min_segment_frames=config.crop_plan_min_segment_frames,
                crop_plan_min_segment_distance_ratio=config.crop_plan_min_segment_distance_ratio,
                crop_plan_smoothing_passes=config.crop_plan_smoothing_passes,
                crop_plan_smooth_window_frames=config.crop_plan_smooth_window_frames,
                output_json=output_folder / "crop_paths.json",
            )
        logs["fixed_16x9_crop"] = list(fixed_crop)

        if config.save_debug:
            with timed_stage("Debug video rendering time", timings):
                render_debug_video(
                    input_video=video_path,
                    output_video=output_folder / "debug_detection.mp4",
                    metadata=metadata,
                    raw_player_detections=raw_players,
                    raw_ball_detections=raw_balls,
                    player_tracks=player_tracks,
                    player_identities=player_identities,
                    active_track_ids=active_track_ids,
                    active_identity_ids=active_identity_ids,
                    ball_track=ball_track,
                    crop_paths=crop_paths,
                    court_bbox=court_bbox,
                    active_court_bbox=active_court_bbox,
                    player_model=config.player_model,
                    ball_model=config.ball_model,
                    player_imgsz=config.player_imgsz,
                    ball_imgsz=config.ball_imgsz,
                )

        with timed_stage("16:9 video rendering time", timings):
            render_fixed_16x9(
                input_video=video_path,
                output_video=output_folder / "main_event_16x9.mp4",
                metadata=metadata,
                crop_box=fixed_crop,
                output_size=config.fixed_output_size,
            )

        with timed_stage("9:16 video rendering time", timings):
            render_dynamic_9x16(
                input_video=video_path,
                output_video=output_folder / "reels_action_9x16.mp4",
                metadata=metadata,
                crop_entries=crop_paths,
                output_size=config.reels_output_size,
            )

        total_elapsed = now_seconds() - total_start
        logs["total_processing_time"] = total_elapsed
        logs["status"] = "success"
        print(f"Total time for {video_path.name}: {total_elapsed:.1f}s")
        write_json(log_path, logs)
    except Exception as exc:
        total_elapsed = now_seconds() - total_start
        logs["total_processing_time"] = total_elapsed
        logs["status"] = "failed"
        logs["errors"].append(
            {
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(log_path, logs)
        print(f"Failed {video_path.name}: {exc}")
        print(f"Total time for {video_path.name}: {total_elapsed:.1f}s")


def main() -> None:
    args = parse_args()
    config = build_config(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    videos = collect_input_videos(config.input_path)
    folder_start = now_seconds()

    if not videos:
        print(f"No supported videos found in {config.input_path}")
        return

    for video_path in videos:
        process_one_video(video_path, config)

    folder_elapsed = now_seconds() - folder_start
    print(f"\nTotal folder processing time: {folder_elapsed:.1f}s")


if __name__ == "__main__":
    main()
