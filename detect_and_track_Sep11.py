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
from lane_detector import calculate_pixels_per_meter

import sys
video_path = sys.argv[1]
lane_width_meters = float(sys.argv[2])

# Create output directory
os.makedirs('outputs', exist_ok=True)

# Calculate pixels per meter from lane detection
print("\n" + "=" * 60)
print("AUTOMATIC CALIBRATION")
print("=" * 60)
pixels_per_meter = calculate_pixels_per_meter(video_path, lane_width_meters)

if pixels_per_meter is None:
    print("ERROR: Calibration failed. Using default value (15.0 px/m).")
    pixels_per_meter = 15.0

print("=" * 60)
print()

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
torch.backends.cudnn.benchmark = True
try:
    model.fuse()
except Exception:
    pass

# 4. Load CLIP model for vehicle classification
print("Loading CLIP model for vehicle classification...")
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name)
clip_processor = CLIPProcessor.from_pretrained(model_name)
clip_model.eval()

yolo_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_device = "cpu"

clip_model.to(clip_device)
print(f"YOLO model will run on {yolo_device}")
print(f"CLIP model loaded on {clip_device}")


# 5. Speed and Color Estimation Setup
FPS = int(cap.get(cv2.CAP_PROP_FPS))
SPEED_SMOOTHING_FRAMES = 5
COLOR_DETECTION_INTERVAL = 5

# Data storage for results
vehicle_data = defaultdict(lambda: {
    'positions': [],
    'speeds': [],
    'colors': [],
    'vehicle_type': None,
    'class_confidence': 0.0,
    'first_seen': 0,
    'last_seen': 0
})

# 6. Define vehicle categories for zero-shot classification
vehicle_categories = [
    "a low-profile car, like a sedan or hatchback",
    "a tall car with a long roof, like an SUV or crossover",
    "a very tall boxy vehicle, like a panel van or minivan",
    "a pickup truck with an open cargo bed in the back"
]

vehicle_map = {
    0: "CAR",
    1: "SUV", 
    2: "VAN",
    3: "PICKUP",
}

def class_group(c):
    if c == 2: return "CAR"
    if c == 3: return "TWO_WHEEL"
    if c in (5, 7): return "HEAVY"
    return "OTHER"

def get_dominant_hsv_color_name(hsv_color):
    h, s, v = hsv_color[0], hsv_color[1], hsv_color[2]

    if v < 60:
        return 'black'
    if v > 190 and s < 40:
        return 'white'
    if s < 45:
        return 'gray'

    if (h >= 0 and h <= 10) or (h >= 170 and h <= 180):
        return 'red'
    elif h <= 25:
        return 'orange'
    elif h <= 35:
        return 'yellow'
    elif h <= 85:
        return 'green'
    elif h <= 130:
        return 'blue'
    elif h <= 155:
        return 'purple'
    elif h <= 169:
        return 'pink'
    else:
        return 'red'

def detect_vehicle_color(image_crop):
    try:
        h, w = image_crop.shape[:2]
        if h < 20 or w < 20:
            return "unknown", 0.0

        y1, y2 = int(h * 0.1), int(h * 0.9)
        x1, x2 = int(w * 0.1), int(w * 0.9)
        center_region_bgr = image_crop[y1:y2, x1:x2]

        if center_region_bgr.size == 0:
            center_region_bgr = image_crop

        hsv_img = cv2.cvtColor(center_region_bgr, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv_img, np.array([0, 0, 20]), np.array([180, 255, 235]))
        pixels = hsv_img[mask > 0]
        
        if pixels.shape[0] < 100:
            return "unknown", 0.0
            
        pixels = pixels.reshape(-1, 3).astype(np.float32)
        total_pixels = pixels.shape[0]

        K = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        counts = np.bincount(labels.flatten())
        
        all_clusters = []
        for i in range(K):
            count = counts[i]
            center_hsv = centers[i].astype(int)
            color_name = get_dominant_hsv_color_name(center_hsv)
            all_clusters.append((count, color_name))

        non_blue_clusters = [cluster for cluster in all_clusters if cluster[1] != 'blue']

        dominant_cluster = None
        if non_blue_clusters:
            dominant_cluster = max(non_blue_clusters, key=lambda x: x[0])
        else:
            dominant_cluster = max(all_clusters, key=lambda x: x[0])

        dominant_count, dominant_name = dominant_cluster
        confidence = dominant_count / float(total_pixels)

        if confidence < 0.15:
             return "unknown", confidence

        return dominant_name, confidence

    except Exception as e:
        print(f"Color detection error: {e}")
        return "unknown", 0.0


def calculate_speed(positions, fps):
    if len(positions) < 2:
        return 0.0
    
    recent_positions = positions[-SPEED_SMOOTHING_FRAMES:] if len(positions) >= SPEED_SMOOTHING_FRAMES else positions
    
    if len(recent_positions) < 2:
        return 0.0
    
    start_pos = recent_positions[0]
    end_pos = recent_positions[-1]
    
    pixel_distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
    
    distance_meters = pixel_distance / pixels_per_meter
    
    time_seconds = (end_pos[2] - start_pos[2]) / fps
    
    if time_seconds == 0:
        return 0.0
    
    speed_kmh = (distance_meters / time_seconds) * 3.6
    
    return speed_kmh

def classify_car(image_crop):
    try:
        rgb_image = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        inputs = clip_processor(
            text=vehicle_categories,
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        
        inputs = {k: v.to(clip_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        confidence = probs.max().item()
        predicted_idx = probs.argmax().item()
        
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
EMA_ALPHA = 0.60
SWITCH_MARGIN = 0.12
BLUR_VAR_MIN = 120.0

# temporal stabilizer for YOLO classes
MIN_FRAMES_ANY = 3
WITHIN_SWITCH_STREAK = 4
GROUP_SWITCH_STREAK = 6
SMALL_AREA_RATIO = 0.012
TRUCK_AR_MIN = 1.6
BUS_AR_MAX = 1.35

# Far-away / small object pruning
MIN_ANALYSIS_AREA = 0.004
RETIRE_TINY_FRAMES = 12
FAR_SHRINK_DROP = 0.35

# freeze-on-approach knobs
AREA_GROWTH_FREEZE = 0.05
APPROACH_FREEZE_FRAMES = 6

frame_count = 0

classification_cache = {}
frame_counter_per_id = {}
track_ema = {}
det_class_state = {}
tiny_counter = {}
best_area = {}
freeze_until = {}

# Main loop
print("Starting processing...")
print()

pbar = tqdm(total=total_frames, desc="Processing video", unit=" frames", 
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

    if len(results) > 0:
        result = results[0]
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            track_id = int(box.id[0]) if box.id is not None else -1
            conf = float(box.conf[0]) if box.conf is not None else 0.0

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            vehicle_data[track_id]['positions'].append((center_x, center_y, frame_count))
            vehicle_data[track_id]['last_seen'] = frame_count
            
            if vehicle_data[track_id]['first_seen'] == 0:
                vehicle_data[track_id]['first_seen'] = frame_count

            bw = max(1, int(x2 - x1))
            bh = max(1, int(y2 - y1))
            area_ratio= (bw * bh) / float(width * height)
            ar = bw / float(bh)
            
            NEAR_AREA_RATIO = 0.04
            if area_ratio > NEAR_AREA_RATIO:
                local_alpha = EMA_ALPHA * 0.4
            else:
                local_alpha = EMA_ALPHA

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
                    track_id in classification_cache and frame_counter_per_id[track_id] % 30 == 0
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
            
            current_speed = calculate_speed(vehicle_data[track_id]['positions'], fps)
            vehicle_data[track_id]['speeds'].append(current_speed)
            
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )
            
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
    
    out.write(frame)
    
    frame_count += 1
    pbar.update(1)
    pbar.set_postfix({'vehicles': len(vehicle_data)})

pbar.close()

cap.release()
out.release()
cv2.destroyAllWindows()

import time
time.sleep(1)

# Save results to CSV
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

df = pd.DataFrame(results_data)
df.to_csv('outputs/vehicle_tracking_results.csv', index=False)

print("\nProcessing complete!")
print(f"Tracked {len(results_data)} vehicles")
print("Output saved to outputs/")