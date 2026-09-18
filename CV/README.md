# Computer Vision Pipeline - ResQmine Rover

Real-time human detection module built with OpenCV for rescue operations.

## Features
- **Real-Time Human Detection**: Uses OpenCV's pre-trained HOG (Histogram of Oriented Gradients) + Linear SVM detector.
- **Terminal Notifications**: Outputs `[INFO] Human detected! (Count: N)` whenever a person is identified in the frame.
- **Visual Alert**: Draws a red bounding box with detection score around every detected human in the live stream.
- **Flexible Source Input**: Switch easily between webcams, recorded video files, or network streams (e.g., ESP32-CAM MJPEG/RTSP stream).
- **Non-Maximum Suppression (NMS)**: Eliminates redundant overlapping boxes.

## Installation

Ensure required dependencies are installed:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Default Webcam
```bash
python3 human_detector.py
```

### 2. Specific Camera Index (e.g., Camera 1)
```bash
python3 human_detector.py --source 1
```

### 3. ESP32-CAM or IP Stream
```bash
python3 human_detector.py --source "http://<ESP32_IP>:81/stream"
```

### 4. Recorded Video File
```bash
python3 human_detector.py --source "path/to/test_video.mp4"
```

### Options
- `--conf`: Detection confidence threshold (default: `0.2`).
- `--nms`: Non-Maximum Suppression overlap threshold (default: `0.4`).
- `--resize-width`: Resize frame width to boost FPS on resource-constrained devices (default: `640`, set `0` for unscaled).

### Controls
- Press `q` or `ESC` in the video window to stop.
- Or press `Ctrl + C` in the terminal.
