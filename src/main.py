"""Webcam object detection demo using YOLOv3-tiny via OpenCV."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np

try:
    import requests
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "requests is required. Install dependencies with 'pip install -r requirements.txt'."
    ) from exc


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
COCO_NAMES_PATH = MODEL_DIR / "coco.names"
FACE_PROTO_PATH = MODEL_DIR / "deploy.prototxt"
FACE_MODEL_PATH = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

YOLO_MODEL_OPTIONS = {
    "nano": (
        "yolov5n.onnx",
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx",
    ),
    "small": (
        "yolov5s.onnx",
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx",
    ),
}
DEFAULT_YOLO_MODEL = "nano"

COCO_NAMES_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
FACE_PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
FACE_MODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index passed to cv2.VideoCapture (default: 0)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.4,
        help="Minimum detection confidence threshold (default: 0.4)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=640,
        help="Square input dimension for YOLO preprocessing (fixed at 640 for bundled model)",
    )
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="Display annotated frames in a window (press 'q' to quit)",
    )
    parser.add_argument(
        "--yolo-model",
        choices=sorted(YOLO_MODEL_OPTIONS.keys()),
        default=DEFAULT_YOLO_MODEL,
        help="YOLO model size to use (nano is fastest, small is more accurate)",
    )
    parser.add_argument(
        "--object-iou",
        type=float,
        default=0.45,
        help="IoU threshold for object detection non-max suppression (default: 0.45)",
    )
    parser.add_argument(
        "--face-min-confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence for faces (default: 0.5)",
    )
    parser.add_argument(
        "--enable-faces",
        action="store_true",
        help="Enable the dedicated face detector (disabled by default)",
    )
    return parser.parse_args()


def ensure_model_file(path: Path, url: str) -> None:
    """Download a model file if it does not exist locally."""
    if path.exists():
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {path.name} ...", flush=True)
    with requests.get(url, stream=True, timeout=90) as response:
        response.raise_for_status()
        with path.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file_obj.write(chunk)
    print(f"Saved {path}")


def letterbox(
    image: np.ndarray,
    size: int,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    new_size = (int(round(width * scale)), int(round(height * scale)))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    pad_w = size - new_size[0]
    pad_h = size - new_size[1]
    pad_left = int(np.floor(pad_w / 2))
    pad_right = int(np.ceil(pad_w / 2))
    pad_top = int(np.floor(pad_h / 2))
    pad_bottom = int(np.ceil(pad_h / 2))

    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)
    return padded, scale, (pad_left, pad_top)


def prepare_networks(yolo_model: str, enable_faces: bool) -> Tuple[cv2.dnn_Net, List[str], Optional[cv2.dnn_Net]]:
    try:
        weights_name, weights_url = YOLO_MODEL_OPTIONS[yolo_model]
    except KeyError as exc:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unsupported YOLO model '{yolo_model}'") from exc

    yolo_weights_path = MODEL_DIR / weights_name

    ensure_model_file(yolo_weights_path, weights_url)
    ensure_model_file(COCO_NAMES_PATH, COCO_NAMES_URL)

    class_names = [line.strip() for line in COCO_NAMES_PATH.read_text().splitlines() if line.strip()]

    object_net = cv2.dnn.readNetFromONNX(str(yolo_weights_path))
    object_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    object_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    face_net: Optional[cv2.dnn_Net] = None
    if enable_faces:
        ensure_model_file(FACE_PROTO_PATH, FACE_PROTO_URL)
        ensure_model_file(FACE_MODEL_PATH, FACE_MODEL_URL)
        face_net = cv2.dnn.readNetFromCaffe(str(FACE_PROTO_PATH), str(FACE_MODEL_PATH))
        face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return object_net, class_names, face_net


def infer_objects(
    net: cv2.dnn_Net,
    class_names: Sequence[str],
    frame: np.ndarray,
    min_confidence: float,
    iou_threshold: float,
    input_size: int,
) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
    padded, scale, (pad_left, pad_top) = letterbox(frame, size=input_size)
    blob = cv2.dnn.blobFromImage(padded, 1 / 255.0, (input_size, input_size), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()  # Shape: (1, 25200, 85)
    detections = outputs[0]

    boxes: List[Tuple[int, int, int, int]] = []
    confidences: List[float] = []
    labels: List[int] = []

    frame_height, frame_width = frame.shape[:2]

    for detection in detections:
        object_conf = float(detection[4])
        if object_conf < min_confidence:
            continue
        class_scores = detection[5:]
        class_id = int(np.argmax(class_scores))
        class_conf = float(class_scores[class_id])
        confidence = object_conf * class_conf
        if confidence < min_confidence:
            continue

        cx, cy, width, height = detection[0:4]
        cx = (cx - pad_left) / scale
        cy = (cy - pad_top) / scale
        width /= scale
        height /= scale

        x = int(max(cx - width / 2, 0))
        y = int(max(cy - height / 2, 0))
        w = int(min(width, frame_width - x))
        h = int(min(height, frame_height - y))
        if w <= 0 or h <= 0:
            continue

        boxes.append((x, y, w, h))
        confidences.append(confidence)
        labels.append(class_id)

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, iou_threshold)
    if len(indices) == 0:
        return []

    results: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
    for idx in indices.flatten():
        x, y, w, h = boxes[idx]
        label = class_names[labels[idx]] if labels[idx] < len(class_names) else f"class_{labels[idx]}"
        results.append((label, confidences[idx], (x, y, x + w, y + h)))
    return results


def infer_faces(
    net: cv2.dnn_Net,
    frame: np.ndarray,
    min_confidence: float,
) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    frame_height, frame_width = frame.shape[:2]
    results: List[Tuple[str, float, Tuple[int, int, int, int]]] = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < min_confidence:
            continue
        x1 = int(detections[0, 0, i, 3] * frame_width)
        y1 = int(detections[0, 0, i, 4] * frame_height)
        x2 = int(detections[0, 0, i, 5] * frame_width)
        y2 = int(detections[0, 0, i, 6] * frame_height)
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, frame_width - 1)
        y2 = min(y2, frame_height - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        results.append(("face", confidence, (x1, y1, x2, y2)))

    return results


def main() -> int:
    args = parse_args()
    if args.min_confidence <= 0 or args.min_confidence >= 1:
        print("--min-confidence expects a value between 0 and 1 (exclusive)", file=sys.stderr)
        return 2
    if args.input_size != 640:
        print("--input-size currently supports only 640 for the bundled YOLOv5s model", file=sys.stderr)
        return 2
    if args.enable_faces and (args.face_min_confidence <= 0 or args.face_min_confidence >= 1):
        print("--face-min-confidence expects a value between 0 and 1 (exclusive)", file=sys.stderr)
        return 2
    if args.object_iou <= 0 or args.object_iou >= 1:
        print("--object-iou expects a value between 0 and 1 (exclusive)", file=sys.stderr)
        return 2

    object_net, class_names, face_net = prepare_networks(args.yolo_model, args.enable_faces)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Unable to open camera index {args.camera}", file=sys.stderr)
        return 1

    print("Press Ctrl+C or 'q' to stop.")
    frame_counter = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera", file=sys.stderr)
                break

            object_results = infer_objects(
                object_net,
                class_names,
                frame,
                args.min_confidence,
                args.object_iou,
                args.input_size,
            )
            face_results: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
            if args.enable_faces and face_net is not None:
                face_results = infer_faces(face_net, frame, args.face_min_confidence)
            result_list = object_results + face_results
            timestamp = time.strftime("%H:%M:%S")
            if result_list:
                print(f"[{timestamp}] Frame {frame_counter}: {len(result_list)} object(s) detected")
                for label, conf, (sx, sy, ex, ey) in result_list:
                    width = max(ex - sx, 0)
                    height = max(ey - sy, 0)
                    print(
                        f"  - {label:<12} confidence={conf:.1%} bbox=({sx},{sy}) {width}x{height}",
                        flush=True,
                    )
            else:
                print(
                    f"[{timestamp}] Frame {frame_counter}: no detections above {args.min_confidence:.0%}",
                    flush=True,
                )

            if args.show_window:
                for label, conf, (sx, sy, ex, ey) in result_list:
                    color = (0, 0, 255) if label == "face" else (0, 255, 0)
                    cv2.rectangle(frame, (sx, sy), (ex, ey), color, 2)
                    text = f"{label}: {conf:.0%}"
                    cv2.putText(
                        frame,
                        text,
                        (sx, max(sy - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2, 
                    )
                cv2.imshow("Vision Object Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Quit signal received from window.")
                    break

            frame_counter += 1
    except KeyboardInterrupt:
        print("Stopping (keyboard interrupt)...")
    finally:
        cap.release()
        if args.show_window:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
