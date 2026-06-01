from __future__ import annotations

from pathlib import Path

import numpy as np

from config import PERSON_CLASS_ID
from video_io import write_json


def _boxes_to_detections(result) -> list[dict]:
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None:
        return []

    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
    classes = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.full(len(xyxy), PERSON_CLASS_ID)
    detections: list[dict] = []
    for box, confidence, class_id in zip(xyxy, confidences, classes):
        x1, y1, x2, y2 = [int(value) for value in box.tolist()]
        detections.append(
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float(confidence),
                "class_id": int(class_id),
                "class_name": "person",
            }
        )
    return detections


def run_player_detection(
    video_path: Path,
    output_json: Path | None,
    model_path: Path,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    half: bool,
) -> dict[int, list[dict]]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    detections_by_frame: dict[int, list[dict]] = {}
    results = model.predict(
        source=str(video_path),
        stream=True,
        classes=[PERSON_CLASS_ID],
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        half=half,
        verbose=False,
    )
    for frame_index, result in enumerate(results):
        detections_by_frame[frame_index] = _boxes_to_detections(result)

    if output_json is not None:
        write_json(
            output_json,
            {
                "model": str(model_path),
                "imgsz": imgsz,
                "conf": conf,
                "class_id": PERSON_CLASS_ID,
                "frames": [
                    {"frame": frame_index, "detections": detections}
                    for frame_index, detections in detections_by_frame.items()
                ],
            },
        )
    return detections_by_frame
