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
SPEED_SMOOTHING_FRAMES = 5
COLOR_DETECTION_INTERVAL = 5

SPEED_BAND_ENABLED = True         # set False to go back to old behavior
SPEED_BAND_FRACTION = 0.10        # 10% of frame height
LANE_ROI_Y_START_RATIO = 0.5      # must match lane_detector.py
LANE_ROI_Y_END_RATIO = 0.9        # must match lane_detector.py

# Will be set after we know frame height
SPEED_BAND_Y1 = None
SPEED_BAND_Y2 = None

# Data storage for results
vehicle_data = defaultdict(lambda: {
    'positions': [],
    'speeds': [],
    'colors': [],
    'vehicle_type': None,
    'class_confidence': 0.0,
    'first_seen': None,
    'last_seen': 0,
    'first_timestamp': None, 
    'last_timestamp': None, 
    'yolo_cls_id': None,      
    'yolo_label': None        
})

# 6. Define vehicle categories for zero-shot classification
vehicle_categories = [
    # 0 - CAR
    "a standard passenger car, sedan or hatchback, low roof and low ground clearance",
    # 1 - SUV
    "a sport-utility vehicle (SUV) or crossover, tall body, high ground clearance, roof rails",
    # 2 - VAN
    "a boxy delivery van or minivan, very tall and rectangular body",
    # 3 - PICKUP
    "a pickup truck with an open cargo bed at the back"
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

def frame_to_timestamp(frame_num, fps):
    """Convert frame number to MM:SS timestamp"""
    total_seconds = int(frame_num / fps)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

def get_dominant_hsv_color_name(hsv_color):
    """
    Map a single HSV value to limited car colors:
    BLACK, WHITE, RED, YELLOW, BLUE.

    h: 0–180, s: 0–255, v: 0–255 (OpenCV HSV)
    """
    h, s, v = int(hsv_color[0]), int(hsv_color[1]), int(hsv_color[2])

    # ---------- BLACK (dark pixels) ----------
    # Very dark -> black
    if v < 55:
        return 'black'

    # Moderately dark + not very colorful -> still black
    if v < 90 and s < 80:
        return 'black'

    # ---------- WHITE / SILVER / GRAY ----------
    # Very bright + low saturation -> white/silver
    if v > 210 and s < 70:
        return 'white'

    # Bright-ish and very low-sat -> white/gray
    if v > 180 and s < 45:
        return 'white'

    # Mid-bright but clearly low-sat -> neutral (white/gray)
    if v > 140 and s < 35:
        return 'white'

    # ---------- Colored regions ----------
    # If saturation is low, treat as neutral (white/gray).
    if s < 45:
        return 'white'

    # RED: include magenta/pink-ish
    if (0 <= h <= 15) or (165 <= h <= 180):
        return 'red'

    # ORANGE / YELLOW
    if 15 < h <= 55:
        return 'yellow'

    # TRUE BLUE (avoid green/cyan)
    if 95 <= h <= 135:
        return 'blue'

    # Anything else odd-ish -> neutral
    return 'white'


def detect_vehicle_color(image_crop):
    try:
        h, w = image_crop.shape[:2]
        if h < 20 or w < 20:
            return "unknown", 0.0

        # ---- 1) ROI: focus on vehicle body, avoid sky & road as much as possible ----
        y1 = int(h * 0.22)   # cut a bit more top (roof/sky reflections)
        y2 = int(h * 0.90)   # keep most of body, avoid bottom road
        x1 = int(w * 0.20)
        x2 = int(w * 0.80)

        roi = image_crop[y1:y2, x1:x2]
        if roi.size == 0:
            roi = image_crop

        # ---- 2) Convert to HSV ----
        hsv_img = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        v_channel = hsv_img[:, :, 2]

        # Keep pixels that aren’t extremely dark or blown out
        mask = (v_channel >= 40) & (v_channel <= 245)
        pixels = hsv_img[mask]

        if pixels.shape[0] < 80:
            # Too few reliable pixels -> bail
            return "unknown", 0.0

        # ---- 3) Sample pixels (for speed) ----
        max_samples = 800
        if pixels.shape[0] > max_samples:
            idx = np.random.choice(pixels.shape[0], max_samples, replace=False)
            sample = pixels[idx]
        else:
            sample = pixels

        # Overall brightness of the car patch
        mean_v = float(sample[:, 2].mean())

        # ---- 4) Vote among palette colors ----
        votes = {
            "black": 0.0,
            "white": 0.0,
            "red":   0.0,
            "yellow":0.0,
            "blue":  0.0,
        }

        for hsv in sample:
            cname = get_dominant_hsv_color_name(hsv)
            if cname in votes:
                votes[cname] += 1.0
            else:
                votes["white"] += 1.0  # shouldn't happen

        total_votes = sum(votes.values())
        if total_votes == 0:
            return "unknown", 0.0

        for c in votes:
            votes[c] /= float(total_votes)

        # ---- 5) Brightness-aware rebalancing (fix white-vs-black) ----
        if mean_v > 150:
            # Overall bright object: strongly favor white over black
            votes["black"] *= 0.35
            votes["white"] *= 1.25
        elif mean_v < 110:
            # Overall darker object: give black a small boost
            votes["black"] *= 1.15

        # ---- 6) Conservative penalty for BLUE so it doesn't dominate accidentally ----
        votes["blue"] *= 0.70  # 30% penalty

        # Re-normalize after adjustments
        total_after = sum(votes.values())
        if total_after > 0:
            for c in votes:
                votes[c] /= total_after

        # ---- 7) Pick dominant color ----
        dominant_color = max(votes, key=votes.get)
        confidence = votes[dominant_color]

        # Require at least ~15% of (sampled) pixels agreeing
        if confidence < 0.15:
            return "unknown", confidence

        # ---- 7a) Guardrail for BLUE (same as before) ----
        if dominant_color == "blue":
            second_best_val = max(v for k, v in votes.items() if k != "blue")
            blue_share = votes["blue"]
            margin = blue_share - second_best_val

            # Only accept BLUE if it clearly dominates
            if blue_share < 0.30 or margin < 0.08:
                # Not clearly blue -> choose neutral (white/black) instead
                if votes["white"] >= votes["black"]:
                    dominant_color = "white"
                    confidence = votes["white"]
                else:
                    dominant_color = "black"
                    confidence = votes["black"]

        # ---- 7b) NEW: Guardrail for BLACK vs WHITE on bright cars ----
        if dominant_color == "black":
            black_share  = votes["black"]
            white_share  = votes["white"]
            margin_bw    = black_share - white_share

            # If the patch is overall bright, it's extremely unlikely to be a black car.
            # mean_v was computed earlier from the same 'sample'.
            if mean_v > 140 and (black_share < 0.35 or margin_bw < 0.05):
                # Treat as white/silver instead
                dominant_color = "white"
                confidence = white_share

        return dominant_color, confidence

    except Exception as e:
        print(f"Color detection error: {e}")
        return "unknown", 0.0

def update_track_color(color_history):
    """
    Track-level color stabilizer.
    BLUE is sticky (paint), WHITE is non-sticky (reflection).
    """
    if not color_history:
        return "unknown"

    recent = color_history[-10:]

    scores = {}
    for c, conf in recent:
        if c == "unknown":
            continue
        scores[c] = scores.get(c, 0.0) + conf

    if not scores:
        return "unknown"

    blue_score  = scores.get("blue", 0.0)
    white_score = scores.get("white", 0.0)
    black_score = scores.get("black", 0.0)

    # ---- KEY RULE ----
    # If blue ever gets reasonable evidence, keep it
    if blue_score >= 0.35:
        return "blue"

    # Otherwise take strongest
    return max(scores, key=scores.get)


   
def calculate_speed(positions, fps):
    """
    Calculate speed using only positions that fall inside the calibrated
    horizontal band (if enabled). This keeps distance roughly at the lane
    width where pixels_per_meter is valid.
    """
    if len(positions) < 2:
        return 0.0

    # Filter positions to those inside the speed band (if enabled)
    if SPEED_BAND_ENABLED and SPEED_BAND_Y1 is not None and SPEED_BAND_Y2 is not None:
        band_positions = [
            p for p in positions
            if SPEED_BAND_Y1 <= p[1] <= SPEED_BAND_Y2   # p = (cx, cy, frame_index)
        ]

        if len(band_positions) < 2:
            return 0.0

        recent_positions = (
            band_positions[-SPEED_SMOOTHING_FRAMES:]
            if len(band_positions) >= SPEED_SMOOTHING_FRAMES
            else band_positions
        )
    else:
        recent_positions = (
            positions[-SPEED_SMOOTHING_FRAMES:]
            if len(positions) >= SPEED_SMOOTHING_FRAMES
            else positions
        )

    if len(recent_positions) < 2:
        return 0.0

    start_pos = recent_positions[0]
    end_pos = recent_positions[-1]

    # pixel distance in image plane
    pixel_distance = np.sqrt(
        (end_pos[0] - start_pos[0])**2 +
        (end_pos[1] - start_pos[1])**2
    )

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
            probs = logits_per_image.softmax(dim=1)  # [1, 4]

        # Convert to numpy
        probs_vec = probs[0].detach().cpu().numpy().astype(np.float32)

        # ---- PRIOR: bias to CAR, down-weight SUV & VAN slightly ----
        # index: 0=CAR, 1=SUV, 2=VAN, 3=PICKUP
        priors = np.array([
            1.1,  # CAR  -> boost
            0.9,  # SUV  -> small penalty
            0.75,  # VAN  -> stronger penalty
            0.95   # PICKUP -> almost neutral
        ], dtype=np.float32)

        probs_vec *= priors
        probs_vec /= probs_vec.sum() + 1e-8  # renormalize

        predicted_idx = int(np.argmax(probs_vec))
        confidence = float(probs_vec[predicted_idx])

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

if SPEED_BAND_ENABLED:
    lane_roi_y_start = int(height * LANE_ROI_Y_START_RATIO)  # same ROI as lane_detector
    lane_roi_y_end   = int(height * LANE_ROI_Y_END_RATIO)

    band_height = int(height * SPEED_BAND_FRACTION)          # 10% of frame
    band_center = (lane_roi_y_start + lane_roi_y_end) // 2   # center of lane ROI

    SPEED_BAND_Y1 = max(0, band_center - band_height // 2)
    SPEED_BAND_Y2 = min(height - 1, band_center + band_height // 2)

    print(f"Speed band (10% of frame) between rows {SPEED_BAND_Y1} and {SPEED_BAND_Y2}")
else:
    SPEED_BAND_Y1 = SPEED_BAND_Y2 = None

# smoothing + quality knobs
EMA_ALPHA = 0.70
SWITCH_MARGIN = 0.08
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
            
            # --- Common padded crop for this track (used for color and CLIP) ---
            padding = 20
            x1_int = max(0, int(x1) - padding)
            y1_int = max(0, int(y1) - padding)
            x2_int = min(width,  int(x2) + padding)
            y2_int = min(height, int(y2) + padding)

            vehicle_crop = frame[y1_int:y2_int, x1_int:x2_int]
            have_good_crop = (
                vehicle_crop.size > 0
                and vehicle_crop.shape[0] > 20
                and vehicle_crop.shape[1] > 20
            )
            
            vehicle_data[track_id]['positions'].append((center_x, center_y, frame_count))
            vehicle_data[track_id]['last_seen'] = frame_count
            vehicle_data[track_id]['last_timestamp'] = frame_to_timestamp(frame_count, fps)
            
            if vehicle_data[track_id]['first_seen'] is None:
                vehicle_data[track_id]['first_seen'] = frame_count
                vehicle_data[track_id]['first_timestamp'] = frame_to_timestamp(frame_count, fps)

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
            
            # --- Store YOLO label for this track (for CSV later) ---
            names = model.names if hasattr(model, "names") else {}
            if isinstance(names, dict):
                name_safe = names.get(resolved_cls, f"cls{resolved_cls}")
            else:
                # just in case names is a list
                try:
                    name_safe = names[resolved_cls]
                except Exception:
                    name_safe = f"cls{resolved_cls}"

            vehicle_data[track_id]['yolo_cls_id'] = resolved_cls
            vehicle_data[track_id]['yolo_label'] = name_safe

            vehicle_type = None
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

                if should_classify and not skip_classify and have_good_crop:
                    gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
                    blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if blur_var < BLUR_VAR_MIN:
                        pass
                    else:
                        vtype_now, vconf_now, raw_now, probs_now = classify_car(vehicle_crop)
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
                            
                            # ---- geometric sanity check for VAN ----
                            # Only allow VAN if it's a fairly tall, big box.
                            # Otherwise treat it as CAR.
                            if candidate_label == "VAN":
                                # ar = width/height
                                # area_ratio = bbox_area / frame_area
                                if (area_ratio < 0.02) or (ar > 2.2):
                                    # too small or too low/long -> more like a car
                                    candidate_label = "CAR"
                                    top1_idx = 0       # CAR index
                                    top1 = float(ema[0])
                                    tmp2 = ema.copy()
                                    tmp2[0] = -1.0
                                    top2 = float(tmp2.max())
                                    margin = top1 - top2



                            prev = classification_cache.get(track_id)
                            prev_label = prev[0] if prev else None
                            prev_conf  = prev[1] if prev else 0.0
                            
                            if (prev_label is None) or (candidate_label == prev_label) or (margin >= SWITCH_MARGIN):
                                vehicle_type = candidate_label
                                class_conf   = top1
                            else:
                                vehicle_type = prev_label
                                class_conf   = prev_conf
                        
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
            
            # --- Color detection for all classes (car, bus, truck, bike) ---
            if (
                not skip_classify
                and have_good_crop
                and (frame_count % COLOR_DETECTION_INTERVAL == 0)
            ):
                color, color_conf = detect_vehicle_color(vehicle_crop)

                if color != "unknown" or len(vehicle_data[track_id]['colors']) == 0:
                    vehicle_data[track_id]['colors'].append((color, color_conf))
                    vehicle_data[track_id]['stable_color'] = update_track_color(
                        vehicle_data[track_id]['colors']
                    )

            # --- Speed calculation for all classes ---
            if SPEED_BAND_ENABLED and SPEED_BAND_Y1 is not None and SPEED_BAND_Y2 is not None:
                in_speed_band = SPEED_BAND_Y1 <= center_y <= SPEED_BAND_Y2
            else:
                in_speed_band = True  # fallback: behave like old code

            if in_speed_band:
                current_speed = calculate_speed(vehicle_data[track_id]['positions'], fps)
                if current_speed > 0:
                    vehicle_data[track_id]['speeds'].append(current_speed)
            # Outside band: don't append zeros; we just reuse last nonzero when drawing
            
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )
            
            # Decide what speed to show on the overlay
            if vehicle_data[track_id]['speeds']:
                # Last valid speed measured inside the band
                display_speed = vehicle_data[track_id]['speeds'][-1]
            else:
                # No valid speed yet
                display_speed = None

            if display_speed is None:
                speed_text = ""   # don't show speed yet
            else:
                speed_text = f" | {display_speed:.1f}km/h"
                
            if resolved_cls == 2 and vehicle_type:
                # ✅ Use stabilized track-level color
                dominant_color = vehicle_data[track_id].get('stable_color', 'unknown')

                label = (
                    f"ID:{track_id} {vehicle_type} {class_conf:.2f} | "
                    f"{dominant_color.upper()}{speed_text}"
                )
            
            else:
                name_safe = vehicle_data[track_id].get('yolo_label', f"cls{resolved_cls}")
                dominant_color = vehicle_data[track_id].get('stable_color', 'unknown')

                if dominant_color != 'unknown':
                    label = (
                        f"ID:{track_id} {name_safe.upper()} | "
                        f"{dominant_color.upper()}{speed_text}"
                    )
                else:
                    label = f"ID:{track_id} {name_safe.upper()}{speed_text}"
            
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

            if SPEED_BAND_ENABLED and SPEED_BAND_Y1 is not None and SPEED_BAND_Y2 is not None:
                cv2.rectangle(
                    frame,
                    (0, SPEED_BAND_Y1),
                    (width - 1, SPEED_BAND_Y2),
                    (255, 0, 0),
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

    # 1) Prefer CLIP type (CAR/SUV/VAN/PICKUP) if present
    clip_type = data.get('vehicle_type')

    # 2) Otherwise fall back to YOLO label (car, bus, truck, motorbike)
    yolo_label = data.get('yolo_label') or "UNKNOWN"

    # What you actually want to show in CSV:
    if clip_type and clip_type != "UNKNOWN":
        csv_type = clip_type
    else:
        csv_type = yolo_label.upper()  # e.g. "BUS", "TRUCK", "MOTORBIKE"

    dominant_color = data.get('stable_color', 'unknown')

    if data['speeds']:
        avg_speed = np.mean(data['speeds'][-10:])
    else:
        avg_speed = 0.0


    
    results_data.append({
        'Vehicle ID': track_id,
        'Type': csv_type,            
        'Color': dominant_color,
        'Speed (km/h)': round(avg_speed, 1),
        'From (MM:SS)': data['first_timestamp'],
        'To (MM:SS)': data['last_timestamp'],
        'Confidence': round(data['class_confidence'], 2),
        'Frames detected': len(data['positions'])
    })


df = pd.DataFrame(results_data)
df.to_csv('outputs/vehicle_tracking_results.csv', index=False)

print("\nProcessing complete!")
print(f"Tracked {len(results_data)} vehicles")
print("Output saved to outputs/")