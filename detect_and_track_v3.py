import cv2
import numpy as np
from ultralytics import YOLO
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm

import sys
video_path = sys.argv[1]
pixels_per_meter = float(sys.argv[2])

# Create output directory
os.makedirs('outputs', exist_ok=True)

cap = cv2.VideoCapture(video_path)

# Check if video opened
if not cap.isOpened():
    print("ERROR: Could not open video file")
    exit()

print(f"Video opened successfully")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {total_frames}")

# 3. Load YOLOv8
model = YOLO("yolov8n.pt")

# Performance optimizations
torch.backends.cudnn.benchmark = True   # Enable cuDNN autotuning for consistent frame size
try:
    model.fuse()  # fuse conv + batchnorm layers (small free speed boost)
except Exception:
    pass

# 4. Load CLIP model for vehicle classification
print("Loading CLIP model for vehicle classification...")
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name)
clip_processor = CLIPProcessor.from_pretrained(model_name)
clip_model.eval()

# We set a device for YOLO (GPU if available) and a separate one for CLIP (CPU)
# This prevents GPU Out-of-Memory crashes.
yolo_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_device = "cpu"  # Force CLIP to CPU to save VRAM

clip_model.to(clip_device)
print(f"YOLO model will run on {yolo_device}")
print(f"CLIP model loaded on {clip_device}")


# 5. Speed and Color Estimation Setup
# Speed estimation parameters
FPS = int(cap.get(cv2.CAP_PROP_FPS))
SPEED_SMOOTHING_FRAMES = 5   # Number of frames to average speed over

# Color detection frequency (every N frames)
COLOR_DETECTION_INTERVAL = 5   # Reduced from 10 to get more samples

# Data storage for results
vehicle_data = defaultdict(lambda: {
    'positions': [],   # List of (x, y, frame) tuples
    'speeds': [],      # List of speed measurements
    'colors': [],      # List of color detections
    'vehicle_type': None,
    'class_confidence': 0.0,
    'first_seen': 0,
    'last_seen': 0
})

# 6.Define vehicle categories for zero-shot classification (NEW PROMPTS)
vehicle_categories = [
    "a low-profile car, like a sedan or hatchback",                  # -> maps to CAR
    "a tall car with a long roof, like an SUV or crossover",        # -> maps to SUV
    "a very tall boxy vehicle, like a panel van or minivan",      # -> maps to VAN
    "a pickup truck with an open cargo bed in the back"             # -> maps to PICKUP
]

# Map to vehicle type
vehicle_map = {
    0: "CAR",
    1: "SUV", 
    2: "VAN",
    3: "PICKUP",
}

# COCO ids used: 2=car, 3=motorcycle, 5=bus, 7=truck
def class_group(c):
    if c == 2: return "CAR"
    if c == 3: return "TWO_WHEEL"
    if c in (5, 7): return "HEAVY"
    return "OTHER"

# --- UPDATED K-MEANS BASED COLOR DETECTION ---

def get_dominant_hsv_color_name(hsv_color):
    """ Helper function to map a single HSV tuple to a color name """
    h, s, v = hsv_color[0], hsv_color[1], hsv_color[2]

    # Check for achromatic colors (white, gray, black)
    # These are defined by low Saturation and high/low Value
    if v < 60:
        return 'black'
    if v > 190 and s < 40: # Higher V, low S
        return 'white'
    if s < 45: # Mid-V, low S
        return 'gray' # Merging silver and gray

    # Chromatic colors (defined by Hue)
    if (h >= 0 and h <= 10) or (h >= 170 and h <= 180):
        return 'red'
    elif h <= 25: # 11-25
        return 'orange'
    elif h <= 35: # 26-35
        return 'yellow'
    elif h <= 85: # 36-85
        return 'green'
    elif h <= 130: # 86-130
        return 'blue'
    elif h <= 155: # 131-155
        return 'purple'
    elif h <= 169: # 156-169
        return 'pink'
    else: # Fallback
        return 'red'

def detect_vehicle_color(image_crop):
    try:
        # 1. Pre-process the image
        h, w = image_crop.shape[:2]
        if h < 20 or w < 20: # Skip tiny crops
            return "unknown", 0.0

        # Focus on the center 80% to avoid edges/windows
        y1, y2 = int(h * 0.1), int(h * 0.9)
        x1, x2 = int(w * 0.1), int(w * 0.9)
        center_region_bgr = image_crop[y1:y2, x1:x2]

        if center_region_bgr.size == 0:
            center_region_bgr = image_crop # Fallback

        # Convert to HSV
        hsv_img = cv2.cvtColor(center_region_bgr, cv2.COLOR_BGR2HSV)

        # 2. Filter pixels (LOOSER MASK)
        # Filter ONLY for extreme shadows (V < 20) and highlights (V > 235).
        mask = cv2.inRange(hsv_img, np.array([0, 0, 20]), np.array([180, 255, 235]))
        pixels = hsv_img[mask > 0]
        
        if pixels.shape[0] < 100: # Not enough valid pixels
            return "unknown", 0.0
            
        pixels = pixels.reshape(-1, 3).astype(np.float32)
        total_pixels = pixels.shape[0]

        # 3. K-Means clustering
        K = 3 # Find the 3 most dominant colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        counts = np.bincount(labels.flatten())
        
        # Get all clusters with their counts and names
        all_clusters = []
        for i in range(K):
            count = counts[i]
            center_hsv = centers[i].astype(int)
            color_name = get_dominant_hsv_color_name(center_hsv)
            all_clusters.append((count, color_name))

        # 4a. Filter out 'blue' clusters
        non_blue_clusters = [cluster for cluster in all_clusters if cluster[1] != 'blue']

        dominant_cluster = None
        if non_blue_clusters:
            # 4b. If we have non-blue clusters, pick the most common one
            dominant_cluster = max(non_blue_clusters, key=lambda x: x[0])
        else:
            # 4c. If ALL clusters were 'blue', then the car must actually be blue.
            # Pick the most common one from the original list.
            dominant_cluster = max(all_clusters, key=lambda x: x[0])

        # 5. Get final name and confidence
        dominant_count, dominant_name = dominant_cluster
        confidence = dominant_count / float(total_pixels)

        if confidence < 0.15:
             return "unknown", confidence

        return dominant_name, confidence

    except Exception as e:
        print(f"Color detection error: {e}")
        return "unknown", 0.0

# --- END NEW COLOR DETECTION ---


def calculate_speed(positions, fps):
    """Calculate speed based on position changes"""
    if len(positions) < 2:
        return 0.0
    
    # Get recent positions for speed calculation
    recent_positions = positions[-SPEED_SMOOTHING_FRAMES:] if len(positions) >= SPEED_SMOOTHING_FRAMES else positions
    
    if len(recent_positions) < 2:
        return 0.0
    
    # Calculate distance between first and last position
    start_pos = recent_positions[0]
    end_pos = recent_positions[-1]
    
    # Euclidean distance in pixels
    pixel_distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
    
    # Convert to meters
    distance_meters = pixel_distance / pixels_per_meter
    
    # Calculate time elapsed
    time_seconds = (end_pos[2] - start_pos[2]) / fps
    
    if time_seconds == 0:
        return 0.0
    
    # Calculate speed in km/h
    speed_kmh = (distance_meters / time_seconds) * 3.6
    
    return speed_kmh

def classify_car(image_crop):
    """Classify car type using CLIP zero-shot classification"""
    try:
        rgb_image = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Process image and text
        inputs = clip_processor(
            text=vehicle_categories,
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        
        # Send inputs to the clip_device (CPU)
        inputs = {k: v.to(clip_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Get highest probability category
        confidence = probs.max().item()
        predicted_idx = probs.argmax().item()
        
        # Also return the full probability vector (as numpy) for smoothing
        probs_vec = probs[0].detach().cpu().numpy()
        
        vehicle_type = vehicle_map[predicted_idx]
        raw_label = vehicle_categories[predicted_idx]
        
        return vehicle_type, confidence, raw_label, probs_vec
    except Exception as e:
        print(f"Classification error: {e}")
        return None, 0.0, "unknown", None

# 6. Prepare video writer
fourcc = cv2.VideoWriter_fourcc(*'avc1')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('outputs/output_tracking.mp4', fourcc, fps, (width, height))

# smoothing + quality knobs
EMA_ALPHA = 0.60         # weight for new probs vs history (0.6 new, 0.4 history)
SWITCH_MARGIN = 0.12     # need at least this top1 - top2 margin to change label
BLUR_VAR_MIN = 120.0     # skip updates when crop is blurrier than this Laplacian variance

# temporal stabilizer for YOLO classes (applies to all tracks)
MIN_FRAMES_ANY = 3          # wait this many frames before locking first class
WITHIN_SWITCH_STREAK = 4    # bus<->truck needs this many consecutive frames to switch
GROUP_SWITCH_STREAK = 6     # changing group (e.g., heavy<->car) needs longer streak
SMALL_AREA_RATIO = 0.012    # only allow group switch if box area is very small (far/uncertain)
TRUCK_AR_MIN = 1.6          # wide aspect ratios bias toward 'truck'
BUS_AR_MAX = 1.35           # tall-ish aspect ratios bias toward 'bus'
# Far-away / small object pruning
MIN_ANALYSIS_AREA   = 0.004   # below this fraction of frame area => skip CLIP classification
RETIRE_TINY_FRAMES  = 12      # remove tracks after this many consecutive tiny frames
FAR_SHRINK_DROP     = 0.35    # consider "far" when area shrinks by 35% from its peak
# freeze-on-approach knobs
AREA_GROWTH_FREEZE = 0.05       # >5% growth vs previous best → approaching fast
APPROACH_FREEZE_FRAMES = 6      # freeze reclassification for these many frames

frame_count = 0

classification_cache = {}   # Stores: {track_id: (vehicle_type, class_conf, manufacturer, manuf_conf)}

frame_counter_per_id = {}   # Track how many frames each ID has been seen

track_ema = {}   # track_id -> np.ndarray of shape (len(vehicle_categories),)

# YOLO detector class stabilizer state (per track_id)
det_class_state = {}   # {track_id: {'frames':int,'current':int|None,'group':str|None,'streak_label':int|None,'streak_len':int}}

# counts how many consecutive "tiny" frames per track
tiny_counter = {}

# stores each track's largest area (for shrink check)
best_area = {}      
freeze_until = {}           # track_id -> frame index until which we freeze       

# 8. Main loop: read frames, detect, track, draw
print("Starting processing...")
print("")

# Initialize progress bar
pbar = tqdm(total=total_frames, desc="Processing video", unit=" frames", 
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 tracking (ByteTrack under the hood)
    results = model.track(
        frame,
        persist=True,
        conf=0.3,
        classes=[2, 3, 5, 7],
        tracker="bytetrack.yaml",
        imgsz=640,
        half=(yolo_device.type == "cuda"),
        device=yolo_device,
        vid_stride=3,
        verbose=False
    )

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

            # Calculate center point for speed estimation
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Store position for speed calculation
            vehicle_data[track_id]['positions'].append((center_x, center_y, frame_count))
            vehicle_data[track_id]['last_seen'] = frame_count
            
            if vehicle_data[track_id]['first_seen'] == 0:
                vehicle_data[track_id]['first_seen'] = frame_count

            # tiny heavy-vehicle filter (before stabilizer)
            bw = max(1, int(x2 - x1))
            bh = max(1, int(y2 - y1))
            area_ratio= (bw * bh) / float(width * height)
            ar = bw / float(bh)
            # Perspective bias correction for near vehicles
            NEAR_AREA_RATIO = 0.04
            if area_ratio > NEAR_AREA_RATIO:
                local_alpha = EMA_ALPHA * 0.4
            else:
                local_alpha = EMA_ALPHA

            # Far distance and retirement logic
            ba = best_area.get(track_id, 0.0)
            if area_ratio > ba:
                best_area[track_id] = area_ratio
                ba = area_ratio

            is_far_now  = (ba > 0.0) and (area_ratio < ba * (1.0 - FAR_SHRINK_DROP))
            is_tiny_now = (area_ratio < MIN_ANALYSIS_AREA)

            cnt = tiny_counter.get(track_id, 0)
            cnt = cnt + 1 if is_tiny_now else 0
            tiny_counter[track_id] = cnt

            skip_classify = is_tiny_now or is_far_now

            if cnt >= RETIRE_TINY_FRAMES:
                classification_cache.pop(track_id, None)
                track_ema.pop(track_id, None)
                best_area.pop(track_id, None)
                tiny_counter.pop(track_id, None)
                frame_counter_per_id.pop(track_id, None)
                det_class_state.pop(track_id, None)
                continue

            if cls_id in (5, 7) and area_ratio < 0.006:
                cls_id = 2

            if track_id not in det_class_state:
                det_class_state[track_id] = {
                    "frames": 0, "current": None, "group": None,
                    "streak_label": None, "streak_len": 0
                }
            st = det_class_state[track_id]
            st["frames"] += 1

            obs_cls = cls_id
            obs_grp = class_group(obs_cls)

            if st["streak_label"] == obs_cls:
                st["streak_len"] += 1
            else:
                st["streak_label"] = obs_cls
                st["streak_len"] = 1

            if st["current"] is None:
                if st["frames"] >= MIN_FRAMES_ANY:
                    st["current"] = obs_cls
                    st["group"] = obs_grp
            else:
                cur_cls = st["current"]
                cur_grp = st["group"]

                if obs_grp == cur_grp:
                    if st["streak_len"] >= WITHIN_SWITCH_STREAK:
                        if cur_grp == "HEAVY":
                            if (obs_cls == 7 and ar >= TRUCK_AR_MIN) or (obs_cls == 5 and ar <= BUS_AR_MAX):
                                st["current"] = obs_cls
                            else:
                                if st["streak_len"] >= WITHIN_SWITCH_STREAK + 2:
                                    st["current"] = obs_cls
                        else:
                            st["current"] = obs_cls
                else:
                    if area_ratio <= SMALL_AREA_RATIO and st["streak_len"] >= GROUP_SWITCH_STREAK:
                        st["current"] = obs_cls
                        st["group"] = obs_grp

            resolved_cls = st["current"] if st["current"] is not None else obs_cls

            vehicle_type = None
            model_name_str = None
            class_conf = 0.0
            
            if resolved_cls == 2:
                if track_id not in frame_counter_per_id:
                    frame_counter_per_id[track_id] = 0
                frame_counter_per_id[track_id] += 1

                prev_best = best_area.get(track_id, 0.0)
                if area_ratio > prev_best:
                    best_area[track_id] = area_ratio

                growth = 0.0
                if prev_best > 0:
                    growth = (area_ratio - prev_best) / max(prev_best, 1e-6)

                if growth > AREA_GROWTH_FREEZE:
                    freeze_until[track_id] = frame_count + APPROACH_FREEZE_FRAMES

                should_classify = (
                    track_id not in classification_cache and frame_counter_per_id[track_id] >= 5
                ) or (
                    track_id in classification_cache and frame_counter_per_id[track_id] % 15 == 0
                )

                if freeze_until.get(track_id, 0) > frame_count:
                    should_classify = False

                if should_classify and not skip_classify:
                    padding = 20
                    x1_int = max(0, int(x1) - padding)
                    y1_int = max(0, int(y1) - padding)
                    x2_int = min(width, int(x2) + padding)
                    y2_int = min(height, int(y2) + padding)
                    
                    car_crop = frame[y1_int:y2_int, x1_int:x2_int]
                    
                    if car_crop.size > 0 and car_crop.shape[0] > 20 and car_crop.shape[1] > 20:
                        gray = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
                        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                        if blur_var < BLUR_VAR_MIN:
                            pass
                        else:
                            vtype_now, vconf_now, raw_now, probs_now = classify_car(car_crop)
                            if probs_now is not None:
                                if track_id not in track_ema:
                                        track_ema[track_id] = probs_now
                                        
                                else:
                                    track_ema[track_id] = local_alpha * probs_now + (1.0 - local_alpha) * track_ema[track_id]
                                
                                ema = track_ema[track_id]
                                top1_idx = int(np.argmax(ema))
                                top1 = float(ema[top1_idx])
                                tmp = ema.copy()
                                tmp[top1_idx] = -1.0
                                top2 = float(tmp.max())
                                margin = top1 - top2

                                candidate_label = vehicle_map[top1_idx]

                                prev = classification_cache.get(track_id)
                                prev_label = prev[0] if prev else None
                                prev_conf  = prev[1] if prev else 0.0
                                
                                if (prev_label is None) or (candidate_label == prev_label) or (margin >= SWITCH_MARGIN):
                                    vehicle_type = candidate_label
                                    class_conf   = top1
                                else:
                                    vehicle_type = prev_label
                                    class_conf   = prev_conf
                            
                            if frame_counter_per_id[track_id] % COLOR_DETECTION_INTERVAL == 0:
                                color, color_conf = detect_vehicle_color(car_crop)
                                if color != "unknown" or len(vehicle_data[track_id]['colors']) == 0:
                                    vehicle_data[track_id]['colors'].append((color, color_conf))
                        
                        old = classification_cache.get(track_id)
                        if old is None:
                            new_tuple = (vehicle_type, class_conf)
                        else:
                            prev_type, prev_conf = old
                            if class_conf > prev_conf:
                                new_tuple = (vehicle_type, class_conf)
                            elif vehicle_type == prev_type:
                                new_tuple = (vehicle_type, 0.5 * (class_conf + prev_conf))
                            else:
                                new_tuple = old

                        classification_cache[track_id] = new_tuple

                if track_id in classification_cache:
                    vehicle_type, class_conf = classification_cache[track_id]
                    vehicle_data[track_id]['vehicle_type'] = vehicle_type
                    vehicle_data[track_id]['class_confidence'] = class_conf
            
            # Calculate current speed
            current_speed = calculate_speed(vehicle_data[track_id]['positions'], fps)
            vehicle_data[track_id]['speeds'].append(current_speed)
            
            # Draw rectangle
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )
            
            # Add text label with speed and color info
            if resolved_cls == 2 and vehicle_type:
                recent_colors = vehicle_data[track_id]['colors']
                if recent_colors:
                    recent_last5 = recent_colors[-5:]
                    valid_colors = [(c, conf) for c, conf in recent_last5 if c != "unknown"]
                    if valid_colors:
                        color_weights = defaultdict(float)
                        for color, conf in valid_colors:
                            color_weights[color] += conf
                        dominant_color = max(color_weights, key=color_weights.get)
                    else:
                        dominant_color = "unknown"
                else:
                    dominant_color = "unknown"
                
                label = f"ID:{track_id} {vehicle_type} {class_conf:.2f} | {dominant_color.upper()} | {current_speed:.1f}km/h"
            else:
                names = model.names if hasattr(model, "names") else {}
                try:
                    name_safe = names[resolved_cls]
                except Exception:
                    name_safe = f"cls{resolved_cls}"
                label = f"ID:{track_id} Class:{name_safe} {conf:.2f} | {current_speed:.1f}km/h"

            
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )
    
    # Write annotated frame
    out.write(frame)
    
    frame_count += 1
    
    # Update progress bar
    pbar.update(1)
    pbar.set_postfix({'vehicles': len(vehicle_data)})

# Close progress bar
pbar.close()

cap.release()
out.release()

# Ensure video file is properly closed and flushed
cv2.destroyAllWindows()
import time
time.sleep(1)

# 9. Save results to CSV
print("\nSaving results to CSV...")
results_data = []

for track_id, data in vehicle_data.items():
    if len(data['positions']) < 5:
        continue
    
    vehicle_type = data['vehicle_type'] or "UNKNOWN"
    
    if data['colors']:
        valid_colors = [(c, conf) for c, conf in data['colors'] if c != "unknown"]
        
        if valid_colors:
            color_weights = defaultdict(float)
            for color, conf in valid_colors:
                color_weights[color] += conf
            
            dominant_color = max(color_weights, key=color_weights.get)
        else:
            dominant_color = "unknown"
    else:
        dominant_color = "unknown"
    
    if data['speeds']:
        avg_speed = np.mean(data['speeds'][-10:])
    else:
        avg_speed = 0.0
    
    duration_frames = data['last_seen'] - data['first_seen']
    duration_seconds = duration_frames / fps
    
    results_data.append({
        'Vehicle ID': track_id,
        'Type': vehicle_type,
        'Color': dominant_color,
        'Speed (km/h)': round(avg_speed, 1),
        'Confidence': round(data['class_confidence'], 2),
        'Duration in seconds': round(duration_seconds, 1),
        'Frames detected': len(data['positions'])
    })

# Create DataFrame and save to CSV
df = pd.DataFrame(results_data)
df.to_csv('outputs/vehicle_tracking_results.csv', index=False)

# END
print("\nProcessing complete!")
print(f"Tracked {len(results_data)} vehicles")
print("Output saved to outputs/")
