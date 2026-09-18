#!/usr/bin/env python3
"""
Human Detection Pipeline for ResQmine Rover
==========================================
Detects humans in a live video stream (webcam, video file, or network camera stream)
using OpenCV's built-in HOG (Histogram of Oriented Gradients) + SVM People Detector.

Features:
- Configurable video source (webcam index, IP stream URL, or video file).
- Terminal alert whenever a human is detected.
- Red bounding box drawn around detected individuals in the live video feed.
- Non-Maximum Suppression (NMS) to eliminate duplicate/overlapping boxes.
- Press 'q' or 'ESC' to exit the video feed.
"""

import argparse
import sys
import cv2
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Real-time Human Detection using OpenCV"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: camera index (e.g., 0), IP stream URL (e.g., http://<ip>:81/stream), or path to video file. Default is '0'.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="Confidence threshold for human detection (default: 0.2).",
    )
    parser.add_argument(
        "--nms",
        type=float,
        default=0.4,
        help="Non-Maximum Suppression threshold (default: 0.4).",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=640,
        help="Resize frame width for faster processing (default: 640, set 0 to keep original).",
    )
    return parser.parse_args()


def get_video_capture(source: str):
    # Determine if source is an integer camera index or a string (URL / file path)
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    return cap


def main():
    args = parse_arguments()

    # 1. Initialize video capture
    print(f"[INFO] Initializing video source: {args.source} ...")
    cap = get_video_capture(args.source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source '{args.source}'.")
        print("[TIP] Check camera connections or stream URL permissions.")
        sys.exit(1)

    # 2. Initialize HOG descriptor with default pre-trained People Detector
    print("[INFO] Loading OpenCV HOG People Detector...")
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())

    print("[INFO] Starting video stream. Press 'q' or 'ESC' to quit.")

    window_name = "ResQmine Rover - Human Detection Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to grab frame or end of stream reached.")
                break

            # Resize frame if requested for improved frame rate
            original_h, original_w = frame.shape[:2]
            scale_ratio = 1.0
            if args.resize_width > 0 and original_w > args.resize_width:
                scale_ratio = args.resize_width / float(original_w)
                dim = (args.resize_width, int(original_h * scale_ratio))
                process_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            else:
                process_frame = frame

            # 3. Detect humans in the frame
            # winStride and padding balance speed vs accuracy
            rects, weights = hog.detectMultiScale(
                process_frame,
                winStride=(4, 4),
                padding=(8, 8),
                scale=1.05,
            )

            boxes = []
            scores = []
            for (x, y, w, h), weight in zip(rects, weights):
                # Scale back coordinates if resized
                if scale_ratio != 1.0:
                    x = int(x / scale_ratio)
                    y = int(y / scale_ratio)
                    w = int(w / scale_ratio)
                    h = int(h / scale_ratio)
                boxes.append([x, y, w, h])
                scores.append(float(weight))

            # Apply Non-Maximum Suppression (NMS) to remove overlapping detections
            indices = []
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(
                    boxes,
                    scores,
                    score_threshold=args.conf,
                    nms_threshold=args.nms,
                )

            # Flatten indices if needed (depends on OpenCV version)
            if len(indices) > 0:
                flattened_indices = [
                    i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
                    for i in indices
                ]
                human_count = len(flattened_indices)
            else:
                human_count = 0

            # 4. Terminal output when human is detected
            if human_count > 0:
                print(f"[INFO] Human detected! (Count: {human_count})")

            # 5. Draw red square (bounding box) around each detected human
            # Color in OpenCV is (Blue, Green, Red) -> (0, 0, 255) for RED
            red_color = (0, 0, 255)

            if human_count > 0:
                for idx in flattened_indices:
                    x, y, w, h = boxes[idx]
                    conf = scores[idx]

                    # Draw red bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), red_color, 2)

                    # Draw label background and text
                    label = f"Human ({conf:.2f})"
                    label_size, _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    top = max(y, label_size[1] + 10)
                    cv2.rectangle(
                        frame,
                        (x, top - label_size[1] - 8),
                        (x + label_size[0] + 6, top),
                        red_color,
                        cv2.FILLED,
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x + 3, top - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

            # Display detection status overlay on top-left of feed
            status_text = (
                f"Status: {human_count} Human(s) Detected"
                if human_count > 0
                else "Status: Searching..."
            )
            status_color = (0, 0, 255) if human_count > 0 else (0, 255, 0)
            cv2.putText(
                frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
            )

            # Show live video feed
            cv2.imshow(window_name, frame)

            # Check for key press: 'q' or 'ESC' (27) to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("[INFO] Quit requested by user.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Stream stopped by user interrupt (Ctrl+C).")
    finally:
        # Clean up resources
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Video capture released and windows destroyed.")


if __name__ == "__main__":
    main()
