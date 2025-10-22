# 1. Imports
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 2. Load video
video_path = "data/RTV.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened
if not cap.isOpened():
    print("ERROR: Could not open video file")
    exit()

print(f"Video opened successfully")
print(f"Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model.to(device)
print(f"CLIP model loaded on {device}")

#6.Define vehicle categories for zero-shot classification
vehicle_categories = [
    "sedan or small compact hatchback passenger car",
    "large SUV or crossover (not a van)",
    "panel van or minivan or cargo van (not a passenger car or suv)",
    "pickup truck with cargo bed"
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
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
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
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

# smoothing + quality knobs
EMA_ALPHA = 0.60          # weight for new probs vs history (0.6 new, 0.4 history)
SWITCH_MARGIN = 0.12      # need at least this top1 - top2 margin to change label
BLUR_VAR_MIN = 120.0      # skip updates when crop is blurrier than this Laplacian variance

# temporal stabilizer for YOLO classes (applies to all tracks)
MIN_FRAMES_ANY = 3           # wait this many frames before locking first class
WITHIN_SWITCH_STREAK = 4     # bus<->truck needs this many consecutive frames to switch
GROUP_SWITCH_STREAK = 6      # changing group (e.g., heavy<->car) needs longer streak
SMALL_AREA_RATIO = 0.012     # only allow group switch if box area is very small (far/uncertain)
TRUCK_AR_MIN = 1.6           # wide aspect ratios bias toward 'truck'
BUS_AR_MAX = 1.35            # tall-ish aspect ratios bias toward 'bus'
# Far-away / small object pruning
MIN_ANALYSIS_AREA   = 0.004   # below this fraction of frame area => skip CLIP classification
RETIRE_TINY_FRAMES  = 12      # remove tracks after this many consecutive tiny frames
FAR_SHRINK_DROP     = 0.35    # consider "far" when area shrinks by 35% from its peak
# freeze-on-approach knobs
AREA_GROWTH_FREEZE = 0.05        # >15% growth vs previous best → approaching fast
APPROACH_FREEZE_FRAMES = 6       # freeze reclassification for these many frames

frame_count = 0

classification_cache = {}  # Stores: {track_id: (vehicle_type, class_conf, manufacturer, manuf_conf)}

frame_counter_per_id = {}  # Track how many frames each ID has been seen

track_ema = {}  # track_id -> np.ndarray of shape (len(vehicle_categories),)

# YOLO detector class stabilizer state (per track_id)
det_class_state = {}  # {track_id: {'frames':int,'current':int|None,'group':str|None,'streak_label':int|None,'streak_len':int}}

# counts how many consecutive "tiny" frames per track
tiny_counter = {}

# stores each track's largest area (for shrink check)
best_area = {}      
freeze_until = {}                # track_id -> frame index until which we freeze       


# 8. Main loop: read frames, detect, track, draw
print("Starting processing...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 tracking (ByteTrack under the hood)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = model.track(
        frame,
        persist=True,
        conf=0.4,
        classes=[2, 3, 5, 7],
        tracker="bytetrack.yaml",
        imgsz=480,
        half=(device == "cuda"),   # only use half precision if GPU available
        device=device,
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

            # tiny heavy-vehicle filter (before stabilizer)
            bw = max(1, int(x2 - x1))
            bh = max(1, int(y2 - y1))
            area_ratio= (bw * bh) / float(width * height)
            ar = bw / float(bh)
            # Perspective bias correction for near vehicles
            # Near (large) vehicles: reduce influence of CLIP reclassification
            # Far (small) vehicles: normal behavior
            NEAR_AREA_RATIO = 0.04  # about 4% of frame area means very close vehicle
            if area_ratio > NEAR_AREA_RATIO:
                # reduce update weight for CLIP smoothing (EMA)
                local_alpha = EMA_ALPHA * 0.4   # weaker update when near
            else:
                local_alpha = EMA_ALPHA

            # Far distance and retirement logic
            # Track best (largest) seen area for this track
            ba = best_area.get(track_id, 0.0)
            if area_ratio > ba:
                best_area[track_id] = area_ratio
                ba = area_ratio

            # Define "far" and "tiny" conditions
            is_far_now  = (ba > 0.0) and (area_ratio < ba * (1.0 - FAR_SHRINK_DROP))
            is_tiny_now = (area_ratio < MIN_ANALYSIS_AREA)

            # Count consecutive tiny frames
            cnt = tiny_counter.get(track_id, 0)
            cnt = cnt + 1 if is_tiny_now else 0
            tiny_counter[track_id] = cnt

            # Skip CLIP classification if far or tiny
            skip_classify = is_tiny_now or is_far_now

            # If it's been tiny for a long time, retire (delete) its state
            if cnt >= RETIRE_TINY_FRAMES:
                classification_cache.pop(track_id, None)
                track_ema.pop(track_id, None)
                best_area.pop(track_id, None)
                tiny_counter.pop(track_id, None)
                frame_counter_per_id.pop(track_id, None)
                det_class_state.pop(track_id, None)
                # Skip further processing for this object
                continue

            # Very small "bus/truck" detections are usually false; treat as car
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

            # consecutive-streak tracker
            if st["streak_label"] == obs_cls:
                st["streak_len"] += 1
            else:
                st["streak_label"] = obs_cls
                st["streak_len"] = 1

            # if we haven't locked anything yet, wait a few frames then lock
            if st["current"] is None:
                if st["frames"] >= MIN_FRAMES_ANY:
                    st["current"] = obs_cls
                    st["group"] = obs_grp
            else:
                cur_cls = st["current"]
                cur_grp = st["group"]

                if obs_grp == cur_grp:
                    # within same group (e.g., bus<->truck): shorter streak + AR bias
                    if st["streak_len"] >= WITHIN_SWITCH_STREAK:
                        if cur_grp == "HEAVY":
                            # bias: wide -> truck, tall-ish -> bus
                            if (obs_cls == 7 and ar >= TRUCK_AR_MIN) or (obs_cls == 5 and ar <= BUS_AR_MAX):
                                st["current"] = obs_cls
                            else:
                                # if prior is weak, require a bit longer streak
                                if st["streak_len"] >= WITHIN_SWITCH_STREAK + 2:
                                    st["current"] = obs_cls
                        else:
                            st["current"] = obs_cls
                else:
                    # cross-group (e.g., HEAVY<->CAR): allow only when tiny and sustained
                    if area_ratio <= SMALL_AREA_RATIO and st["streak_len"] >= GROUP_SWITCH_STREAK:
                        st["current"] = obs_cls
                        st["group"] = obs_grp

            resolved_cls = st["current"] if st["current"] is not None else obs_cls


            # Only classify cars (class 2) into car/suv/van/pickup
            vehicle_type = None
            model_name_str = None
            class_conf = 0.0
            
            if resolved_cls == 2:  # Car class - needs sub-classification
                # Track frame count for this ID
                if track_id not in frame_counter_per_id:
                    frame_counter_per_id[track_id] = 0
                frame_counter_per_id[track_id] += 1

                # --- Approach detection & freeze window ---
                prev_best = best_area.get(track_id, 0.0)
                if area_ratio > prev_best:
                    best_area[track_id] = area_ratio

                growth = 0.0
                if prev_best > 0:
                    growth = (area_ratio - prev_best) / max(prev_best, 1e-6)

                # if area is exploding (vehicle rushing toward camera), freeze reclass
                if growth > AREA_GROWTH_FREEZE:
                    freeze_until[track_id] = frame_count + APPROACH_FREEZE_FRAMES

                # Only classify after seeing the car for 5 frames (delay)
                # Then reclassify every 10 frames to improve accuracy
                should_classify = (
                    track_id not in classification_cache and frame_counter_per_id[track_id] >= 5
                ) or (
                    track_id in classification_cache and frame_counter_per_id[track_id] % 15 == 0
                )

                # apply freeze window
                if freeze_until.get(track_id, 0) > frame_count:
                    should_classify = False

                if should_classify and not skip_classify:
                    # Crop the car region with padding
                    padding = 20  # Add 20 pixels padding on each side
                    x1_int = max(0, int(x1) - padding)
                    y1_int = max(0, int(y1) - padding)
                    x2_int = min(width, int(x2) + padding)
                    y2_int = min(height, int(y2) + padding)
                    
                    car_crop = frame[y1_int:y2_int, x1_int:x2_int]
                    
                    # Classify only if crop is valid
                    if car_crop.size > 0 and car_crop.shape[0] > 20 and car_crop.shape[1] > 20:
                        # Skip blurry frames to avoid degrading the EMA
                        gray = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
                        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                        if blur_var < BLUR_VAR_MIN:
                             # Do not update anything on this frame
                             pass
                        else:
                            vtype_now, vconf_now, raw_now, probs_now = classify_car(car_crop)
                            if probs_now is not None:
                                # EMA smoothing of probabilities
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

                                # Shape-based adjustment
                                if ar < 1.3 and candidate_label == "VAN":
                                    candidate_label = "CAR"
                                elif ar > 1.6 and candidate_label == "CAR":
                                    candidate_label = "SUV"

                                # Previous label (if any)
                                prev = classification_cache.get(track_id)
                                prev_label = prev[0] if prev else None
                                prev_conf  = prev[1] if prev else 0.0
                                
                                # Hysteresis: only switch labels when margin is strong
                                if (prev_label is None) or (candidate_label == prev_label) or (margin >= SWITCH_MARGIN):
                                    # accept/update
                                    vehicle_type = candidate_label
                                    class_conf   = top1
                                else:
                                    # keep previous (avoid flip-flop on low margin)
                                    vehicle_type = prev_label
                                    class_conf   = prev_conf
                        
                        # Merge with previous (if any) and then write once
                        old = classification_cache.get(track_id)
                        if old is None:
                            new_tuple = (vehicle_type, class_conf)
                        else:
                            prev_type, prev_conf = old
                            if class_conf > prev_conf:
                                # take the stronger new prediction
                                new_tuple = (vehicle_type, class_conf)
                            elif vehicle_type == prev_type:
                                # same type: smooth the confidence, keep stronger manufacturer
                                new_tuple = (vehicle_type, 0.5 * (class_conf + prev_conf))
                            else:
                                # different type but weaker margin: keep old (hysteresis)
                                new_tuple = old

                        classification_cache[track_id] = new_tuple


                # Get classification from cache
                if track_id in classification_cache:
                    vehicle_type, class_conf = classification_cache[track_id]
            
            # Draw rectangle
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),  # Green box
                2
            )
            
            # Add text label
            if resolved_cls == 2 and vehicle_type:
                label = f"ID:{track_id} {vehicle_type} {class_conf:.2f}"
            else:
                names = model.names if hasattr(model, "names") else {}
                try:
                    name_safe = names[resolved_cls]
                except Exception:
                    name_safe = f"cls{resolved_cls}"
                label = f"ID:{track_id} Class:{name_safe} {conf:.2f}"

            
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