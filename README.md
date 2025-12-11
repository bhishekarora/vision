# Vision Object Detection 

This is a minimal OpenCV project that opens your default webcam, runs a YOLOv5 nano object detector (with an optional face detector), and prints the top detections for every frame to the console. The project lazily downloads the required model files the first time you run it.

## Prerequisites

- Python 3.9 or newer
- A working webcam
- `pip` for installing Python packages

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

Run the detector with:

```bash
python -m src.main
```

By default the script keeps running until you press `Ctrl+C`. Each frame is processed synchronously; detections are printed to the terminal including class name, confidence, and bounding box size. Use `--camera <index>` to switch to a different webcam (USB cams are often at index `1` or higher). A lightweight display window is optional and can be toggled with the `--show-window` flag.

Faces are highlighted in red (SSD face detector) while general objects rely on YOLOv5 (green). Adjust thresholds with `--min-confidence`, `--face-min-confidence`, the YOLO IoU suppression via `--object-iou`, and keep the YOLO input size at `--input-size 640` (fixed for the bundled model). Choose between the fast `--yolo-model nano` (default) and the more accurate `--yolo-model small`.
Faces are disabled by default; enable them with `--enable-faces` when you need them.

### Performance tips
- Stick with the `nano` model for best CPU throughput; only switch to `--yolo-model small` if you need higher accuracy.
- Keep faces disabled unless required (`--enable-faces`), because the SSD face model adds extra inference time.
- Ensure nothing else is accessing the camera so frames arrive without delay; you can also set `--min-confidence` a bit higher to suppress noisy detections sooner.
- If your hardware supports it, set `OPENCV_DNN_TARGET=CUDA` or `OPENCV_DNN_TARGET=OPENCL` before running to leverage GPU or OpenCL acceleration (requires OpenCV built with those backends).
- If you export a custom YOLOv5/v8 ONNX model with dynamic input support you can relax the 640 restriction and experiment with smaller sizes for extra speed.

## Repository Sync Script

`scripts/sync.sh` streamlines pushing code to a GitHub repository:

```bash
bash scripts/sync.sh "Optional commit message"
```

On the first run it will:

1. Initialize a Git repository if needed.
2. Add the remote `origin` pointing at `git@github.com:bhishekarora/vision.git` (override by setting `SYNC_REMOTE`).
3. Stage all tracked and untracked changes.
4. Commit with either the provided commit message or an auto-generated timestamped message.
5. Push to the `main` branch.

Subsequent runs reuse the existing configuration. If there are no local changes, the script exits without committing.

## Notes

- The first execution downloads ~35 MB of model files (YOLOv5n ONNX + SSD face detector) into the `models/` directory; enabling faces adds ~25 MB.
- To run headless (no OpenCV window), omit the `--show-window` option.
- Stop the script promptly if you encounter camera access issues to release the webcam handle.
