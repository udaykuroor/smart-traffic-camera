# 1. Imports
import cv2
import numpy as np
from ultralytics import YOLO

# 2. Load video
video_path = "data/RTV.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened
if not cap.isOpened():
    print("ERROR: Could not open video file")
    exit()

print(f"✓ Video opened successfully")
print(f"Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

# 3. Load YOLOv8
model = YOLO("yolov8n.pt")

# 4. Prepare video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

frame_count = 0

# 5. Main loop: read frames, detect, track, draw
print("Starting processing...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 tracking (ByteTrack under the hood)
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    
    # Get result for the current frame
    if len(results) > 0:
        result = results[0]
        # Loop through each detected object
        for box in result.boxes:
            # Get coordinates, class, and ID
            x1, y1, x2, y2 = box.xyxy[0]
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            track_id = int(box.id[0]) if box.id is not None else -1
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            
            # Draw rectangle
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),  # Green box
                2
            )
            
            # Add text label: ID + class + confidence
            label = f"ID:{track_id} Class:{model.names[cls_id]} {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),  # Yellow text
                2
            )
    
    # Write annotated frame
    out.write(frame)
    
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
print(f"Done. Processed {frame_count} frames. Output saved to 'output_video.mp4'")