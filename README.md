# Sports Auto-Framing Pipeline

This is a self-implemented sports video auto-framing project for pickleball highlight clips. It combines classical computer vision, YOLO object detection, ball tracking, and crop-path generation to create fixed landscape videos and vertical reels-style videos from raw court footage.

The current version is tuned around pickleball, but the project is designed as a reusable auto-framing pipeline. In the future, the same structure can be adapted to another sport by changing the detection model, class filters, court or field detection logic, and crop rules.

The project uses pretrained YOLO models only. It does not train a custom model and does not require manual court marking.

## Pipeline

The pipeline can process either one video file or a folder of videos:

```text
input video or input folder
-> read video metadata
-> detect a fixed court bounding box with classical OpenCV
-> run YOLO player/person detection
-> run YOLO sports-ball detection
-> build player tracks and player identities
-> select active players from whole-video movement inside the court
-> validate raw ball detections
-> build a cleaned ball track with short missing-frame prediction
-> generate one fixed 16:9 crop from the court and active players
-> generate one dynamic 9:16 crop from the validated ball path
-> render a debug video
-> render final MP4 outputs
-> save JSON logs and intermediate results
```

Court detection is classical computer vision only. It samples frames, builds a static background, detects line-like court features, and estimates one rough court bounding box for the whole video.

Player detection uses a YOLO model filtered to the person class. Since YOLO can also detect spectators, the project does not use raw person boxes directly. It builds whole-video player identities and keeps the most active players based on movement inside the active court area.

Ball detection uses a separate YOLO model filtered to the sports-ball class. Raw ball detections are validated before they affect the crop. The current 9:16 crop follows the validated ball path, because this performed better than blending the crop target with active player positions in the tested pickleball clips.

The fixed 16:9 output is stable. It does not move during the video. The dynamic 9:16 output moves smoothly and is intended for short-form social media viewing.

## Project Structure

```text
main.py
```

Main CLI entry point. It parses arguments, builds the pipeline config, processes one video or a folder, prints timing, and saves per-video logs.

```text
config.py
```

Shared constants and dataclass configuration, including supported extensions, YOLO class IDs, crop settings, and tracking thresholds.

```text
video_io.py
```

Video metadata loading, input path normalization, input video collection, OpenCV writer creation, FFmpeg-based MP4 encoding, and JSON writing.

```text
court_detection.py
```

Classical court detection. It builds a median/static background image from sampled frames, detects edges and Hough lines, filters court-like line segments, and estimates one fixed court bounding box.

```text
yolo_player_detection.py
yolo_ball_detection.py
```

YOLO inference wrappers. Player detection uses the person class. Ball detection uses the sports-ball class. The model paths, image sizes, confidence thresholds, IoU thresholds, device, and half precision are configurable.

```text
player_tracking.py
player_identity.py
active_player_filter.py
```

Player post-processing. These modules connect frame-level person detections into track fragments, merge fragments into player identities, and choose active players using whole-video movement inside the court.

```text
ball_validation.py
ball_tracking.py
```

Ball post-processing. These modules reject obvious false ball detections, score candidates, track the ball over time, and briefly predict ball position when detections are missing.

```text
crop_utils.py
crop_path_generation.py
```

Crop geometry and crop planning. The fixed 16:9 crop is based on the court and active players. The dynamic 9:16 crop is based on the validated ball path, with smoothing and movement limits.

```text
render_debug.py
render_fixed_16x9.py
render_dynamic_9x16.py
```

Video rendering modules. The debug video overlays detections, court boxes, crop previews, and ball status. The fixed and dynamic renderers create the final cropped videos.

```text
timing_utils.py
```

Simple timing helpers for printing and saving stage runtime.

## Install

Clone the `pickleball-yolo` branch and enter the project folder:

```bash
git clone -b pickleball-yolo https://github.com/kwanyinsan/sports-auto-framing.git
cd sports-auto-framing
```

Create a conda environment:

```bash
conda create -n sports-autoframe python=3.10 -y
conda activate sports-autoframe
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Install FFmpeg and make sure it is available from the command line:

```bash
ffmpeg -version
```

FFmpeg is required because the final outputs are encoded as H.264/AAC MP4 files.

On Linux, install FFmpeg with your package manager, for example:

```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

## GPU

The project can run on CPU, but YOLO video inference is much more practical with an NVIDIA GPU.

Tested baseline:

```text
torch 2.5.1+cu121
CUDA 12.1
```

Recommended minimum for this project:

```text
torch >= 2.5.1
torchvision >= 0.20.1
```

Choose the PyTorch install command that matches your own GPU, CUDA runtime, and operating system from the official PyTorch install selector:

https://pytorch.org/get-started/locally/

Example CUDA 12.1 install:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

After installing PyTorch, verify GPU access:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

If `torch.cuda.is_available()` prints `True`, run the project with:

```text
--device 0 --half true
```

If GPU memory is not enough, reduce `--ball_imgsz` first. The ball model usually benefits most from a larger image size, but it is also the part most likely to use more memory.

## Models

The project uses two separate pretrained YOLO model files:

```text
player model: detects person/player
ball model: detects sports ball/pickleball candidate
```

Example:

```bash
--player_model models/yolo26l.pt
--ball_model models/yolo26x.pt
```

The model files are not created by this project. Put your pretrained weights in `models/` or pass the full path to the weight file.

This project is not a custom trained pickleball detector. It uses pretrained YOLO object classes and rule-based validation/tracking. For another sport, the likely replacement points are:

- model weights
- YOLO class filters
- ball validation rules
- court/field detection logic
- crop-path logic

## Running

Run one video:

```bash
python main.py \
  --input_video "highlights/_Concatenated__1_Highlights (4).mp4" \
  --output_dir results_single \
  --player_model models/yolo26l.pt \
  --ball_model models/yolo26x.pt \
  --player_imgsz 960 \
  --ball_imgsz 1280 \
  --device 0 \
  --half true \
  --save_debug true
```

Run a folder:

```bash
python main.py \
  --input highlights \
  --output_dir results_batch \
  --player_model models/yolo26l.pt \
  --ball_model models/yolo26x.pt \
  --player_imgsz 960 \
  --ball_imgsz 1280 \
  --device 0 \
  --half true \
  --save_debug true
```

Linux path example:

```bash
python main.py \
  --input_video "/home/user/videos/clip.mp4" \
  --output_dir "/home/user/results" \
  --player_model "/home/user/models/yolo26l.pt" \
  --ball_model "/home/user/models/yolo26x.pt" \
  --player_imgsz 960 \
  --ball_imgsz 1280 \
  --device 0 \
  --half true \
  --save_debug true
```

Use exactly one input argument:

```text
--input
--input_video
--input_dir
```

`--input` can be either a video file or a folder. `--input_video` is explicit single-video mode. `--input_dir` is kept for folder-mode compatibility.

## Arguments

```text
--input
```

Input video file or input folder. This is the most flexible input argument.

```text
--input_video
```

Single input video file.

```text
--input_dir
```

Input folder containing videos. This is kept for compatibility with the original folder-mode command.

```text
--output_dir
```

Root folder where per-video result folders are created.

```text
--player_model
```

Path to the YOLO model used for player/person detection.

```text
--ball_model
```

Path to the YOLO model used for ball/sports-ball detection.

```text
--player_imgsz
```

YOLO inference image size for player detection. Larger values may improve detection but cost more time and memory.

```text
--ball_imgsz
```

YOLO inference image size for ball detection. The ball is small, so this often benefits from a larger value such as `1280`, if the GPU can handle it.

```text
--device
```

YOLO inference device. Use `0` for the first CUDA GPU or `cpu` for CPU.

```text
--half
```

Use half precision during YOLO inference. Usually set to `true` for CUDA GPU and `false` for CPU.

```text
--player_conf
```

Confidence threshold for raw player/person detections.

```text
--ball_conf
```

Confidence threshold for raw ball detections. This can be lower than player confidence because the ball is small and harder to detect.

```text
--player_iou
```

YOLO IoU threshold for player detection non-max suppression.

```text
--ball_iou
```

YOLO IoU threshold for ball detection non-max suppression.

```text
--active_players
```

Maximum number of active player identities to keep. `4` is useful for doubles-style clips.

```text
--active_court_shrink
```

Shrinks the detected court area for active-player foot-point checks. The current implementation preserves the court's long axis and shrinks mainly the short axis.

```text
--min_player_track_frames
```

Minimum visible frames required before a player identity can be considered active.

```text
--min_player_inside_ratio
```

Minimum ratio of frames where the player's foot point is inside the active court box.

```text
--min_player_movement_ratio
```

Minimum movement inside the active court, measured as a ratio of the frame diagonal. This helps reject people who stand still near the court.

```text
--identity_merge_max_gap_frames
```

Maximum frame gap allowed when merging broken player track fragments into one identity.

```text
--identity_merge_score_threshold
```

Minimum merge score required to combine two player track fragments.

```text
--identity_merge_max_distance_ratio
```

Maximum merge distance as a ratio of the frame diagonal.

```text
--ball_min_conf_for_crop
```

Minimum ball confidence used when deciding whether a raw ball detection can influence the crop.

```text
--ball_track_score_threshold
```

Minimum validated ball score for a ball position to be treated as reliable.

```text
--max_ball_missing_frames
```

Number of frames the ball tracker may briefly predict after missing detections.

```text
--crop_smoothing
```

Compatibility smoothing value kept in config/logs. The current offline 9:16 path mainly uses centered smoothing and movement limits.

```text
--max_crop_move_px
```

Maximum crop-center movement per frame for the dynamic 9:16 crop.

```text
--crop_deadzone_ratio
```

Compatibility argument kept in config/logs. Earlier crop logic used this for center dead-zone behavior.

```text
--ball_crop_weight
```

Compatibility argument from the earlier player-aware crop mode. The current main 9:16 method follows the validated ball path.

```text
--predicted_ball_crop_weight
```

Compatibility argument from the earlier player-aware crop mode.

```text
--startup_player_lock_sec
```

Compatibility argument retained for existing command shapes.

```text
--player_keep_margin_ratio
```

Compatibility argument retained for existing command shapes.

```text
--max_ball_offset_ratio
```

Compatibility argument retained for existing command shapes.

```text
--focus_switch_angle_deg
```

Compatibility argument retained for existing command shapes.

```text
--focus_switch_min_frames
```

Used as part of ball-point cleaning and temporal grouping.

```text
--focus_player_search_frames
```

Used as part of ball-point cleaning and temporal grouping. Earlier player-aware crop logic also used it to search for a nearby active player.

```text
--crop_plan_min_ball_score
```

Minimum validated ball score used when building the offline 9:16 crop path.

```text
--crop_plan_min_segment_frames
```

Compatibility argument from rally-segment planning. Kept for command compatibility.

```text
--crop_plan_min_segment_distance_ratio
```

Compatibility argument from rally-segment planning. Kept for command compatibility.

```text
--crop_plan_smoothing_passes
```

Number of centered smoothing passes applied to the 9:16 crop target path.

```text
--crop_plan_smooth_window_frames
```

Centered smoothing window size in frames.

```text
--court_padding
```

Padding added around the detected court box.

```text
--event_padding
```

Padding added around the fixed 16:9 event area.

```text
--reels_output_width
--reels_output_height
```

Output size for the vertical reels video. Default is `1080x1920`.

```text
--fixed_output_width
--fixed_output_height
```

Output size for the fixed landscape video. Default is `1920x1080`.

```text
--full_frame_court_limit_sec
```

For short videos under this duration, court detection can use all frames. Longer videos use sampled frames.

```text
--court_resize_width
```

Resize width used for court detection analysis.

```text
--long_video_court_sample_count
```

Maximum number of frames used for court detection on longer videos.

```text
--save_debug
```

Whether to render `debug_detection.mp4`.

## Input And Output

The project accepts:

```text
.mp4
.mov
.avi
.mkv
```

For each input video, the pipeline creates one output folder named after the video file stem.

Example:

```text
results_batch/
  match_001/
    court_result.json
    raw_detections.json
    player_tracks.json
    player_identities.json
    active_players.json
    ball_track.json
    crop_paths.json
    debug_detection.mp4
    main_event_16x9.mp4
    reels_action_9x16.mp4
    logs.json
```

Output file meanings:

```text
court_result.json
```

Detected fixed court bounding box, active court box, court detection method, confidence, and video metadata.

```text
raw_detections.json
```

Raw YOLO player/person detections and raw YOLO ball detections per frame.

```text
player_tracks.json
```

Short player track fragments built from raw person detections.

```text
player_identities.json
```

Merged player identities built from track fragments across the whole video.

```text
active_players.json
```

Selected active player identities. These are chosen mainly from movement inside the active court area.

```text
ball_track.json
```

Validated ball track with detected, predicted, or missing status per frame.

```text
crop_paths.json
```

Per-frame fixed 16:9 crop and dynamic 9:16 crop coordinates, including crop mode and ball status.

```text
debug_detection.mp4
```

Full-frame debug video with court boxes, detections, active players, ball status, crop previews, and crop path.

```text
main_event_16x9.mp4
```

Final fixed landscape output. The crop does not move.

```text
reels_action_9x16.mp4
```

Final vertical output. The crop follows the validated ball path.

```text
logs.json
```

Runtime configuration, timing for each major stage, status, errors if any, and video metadata.

Final videos are written as:

```text
container: MP4
video codec: H.264 / AVC / avc1
pixel format: yuv420p
audio codec: AAC when source audio exists
faststart: enabled
```

This avoids the common issue where a file has an `.mp4` extension but uses a less compatible codec.
