import cv2
import numpy as np

def detect_lane_width_pixels(video_path, num_samples=10):
    """
    Detect lane width in pixels from video
    
    Args:
        video_path: Path to video file
        num_samples: Number of frames to sample for measurement
    
    Returns:
        average_lane_width_pixels: Average lane width in pixels
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("ERROR: Could not open video")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video resolution: {frame_width}x{frame_height}")
    print(f"Sampling {num_samples} frames for lane detection...")
    
    # Sample frames evenly throughout the video
    sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    
    lane_widths = []
    
    for idx, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Focus on lower half of frame (where road lanes are)
        roi_y_start = int(frame_height * 0.5)
        roi_y_end = int(frame_height * 0.6)
        edges_roi = edges[roi_y_start:roi_y_end, :]
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges_roi,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=100,
            maxLineGap=50
        )
        
        if lines is None:
            continue
        
        # Filter for near-vertical lines (lane markings)
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle from horizontal
            if x2 - x1 == 0:
                angle = 90
            else:
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Keep nearly vertical lines (60-120 degrees from horizontal)
            if 60 < angle < 120:
                # Store x-coordinate (horizontal position) at middle of line
                vertical_lines.append((x1 + x2) / 2)
        
        if len(vertical_lines) < 2:
            continue
        
        # Remove duplicates (lines very close together)
        vertical_lines = sorted(set(vertical_lines))
        unique_lines = []
        for i, x in enumerate(vertical_lines):
            if i == 0 or x - unique_lines[-1] > frame_width * 0.05:  # At least 5% apart
                unique_lines.append(x)
        
        # Calculate distances between consecutive lines
        distances = []
        for i in range(len(unique_lines) - 1):
            dist = unique_lines[i + 1] - unique_lines[i]
            # Filter reasonable lane widths (15% to 50% of frame width)
            if frame_width * 0.15 < dist < frame_width * 0.5:
                distances.append(dist)
        
        if distances:
            # Use median distance for this frame
            lane_widths.append(np.median(distances))
            print(f"  Frame {idx+1}/{num_samples}: Detected {len(unique_lines)} lane lines, width: {np.median(distances):.1f}px")
    
    cap.release()
    
    if not lane_widths:
        # Fallback: use heuristic (assume lane is ~25% of frame width)
        print("⚠ Warning: Could not detect lanes reliably. Using fallback estimate (25% of frame width).")
        fallback_width = frame_width * 0.25
        print(f"  Fallback lane width: {fallback_width:.1f} pixels")
        return fallback_width
    
    # Return median of all measurements for robustness
    final_width = np.median(lane_widths)
    print(f"✓ Lane detection successful! Median width across {len(lane_widths)} samples: {final_width:.1f} pixels")
    
    return final_width


def calculate_pixels_per_meter(video_path, lane_width_meters=3.65):
    """
    Calculate pixels per meter calibration value
    
    Args:
        video_path: Path to video file
        lane_width_meters: Real-world lane width in meters
    
    Returns:
        pixels_per_meter: Calibration value
    """
    print(f"Detecting lane width from video...")
    print(f"Target lane width: {lane_width_meters} meters")
    print()
    
    lane_width_pixels = detect_lane_width_pixels(video_path)
    
    if lane_width_pixels is None:
        print("ERROR: Could not detect lane width")
        return None
    
    pixels_per_meter = lane_width_pixels / lane_width_meters
    
    print()
    print("=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"Lane width (pixels): {lane_width_pixels:.1f} px")
    print(f"Lane width (meters): {lane_width_meters} m")
    print(f"Pixels per meter: {pixels_per_meter:.2f} px/m")
    print("=" * 60)
    
    return pixels_per_meter


# Test function
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python lane_detector.py <video_path> [lane_width_meters]")
        sys.exit(1)
    
    video = sys.argv[1]
    lane_width = float(sys.argv[2]) if len(sys.argv) > 2 else 3.65
    
    ppm = calculate_pixels_per_meter(video, lane_width)
    if ppm:
        print(f"\n✓ Success! Use pixels_per_meter = {ppm:.2f}")
    else:
        print("\n✗ Failed to calculate calibration")