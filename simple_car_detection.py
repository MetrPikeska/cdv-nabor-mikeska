import cv2
from ultralytics import YOLO
import torch
import json
from shapely.geometry import Point, Polygon, LineString
import csv
import os
from collections import defaultdict

print("CUDA available:", torch.cuda.is_available())

def detect_cars(video_path, model_path, roi_path, exit_lines_path, output_csv):
    """
    Event-based car detection with aggregation per minute.
    
    Key improvements:
    - Each vehicle is counted only once per exit line (event-based)
    - Results are aggregated by minute
    - Uses bottom-center point of bounding box for stability
    - Filters out short tracks and minimal movements
    - Handles track loss gracefully
    """
    
    # Load YOLO model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(model_path)
    model.to(device)

    # Load ROI polygon
    with open(roi_path, 'r') as f:
        roi_data = json.load(f)
    roi_polygon = Polygon(roi_data[0])

    # Load exit lines
    with open(exit_lines_path, 'r') as f:
        exit_lines_data = json.load(f)
    exit_lines = {key: LineString(value) for key, value in exit_lines_data.items()}

    # Load exclusion polygon (if any)
    exclusion_polygon = None
    exclusion_path = "output/exclusion.json"
    if os.path.exists(exclusion_path):
        with open(exclusion_path, 'r') as f:
            exclusion_data = json.load(f)
        exclusion_polygon = Polygon(exclusion_data[0])

    # Generate unique output CSV filename
    attempt = 1
    base_output_csv = output_csv
    while os.path.exists(output_csv):
        output_csv = base_output_csv.replace('.csv', f'_attempt{attempt}.csv')
        attempt += 1

    # Open video to get FPS and frame count
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Default fallback
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # State management for tracking
    track_state = {}  # track_id -> {'detected_in_roi': bool, 'exits_crossed': {exit_id: bool}, 'frame_count': int, 'positions': []}
    crossing_counts = defaultdict(lambda: defaultdict(int))  # minute -> exit_id -> count
    inactive_tracks = {}  # track_id -> frames_inactive
    
    TRACK_TIMEOUT = 30  # frames
    MIN_TRACK_LENGTH = 5  # minimum frames to be considered valid
    MIN_MOVEMENT = 10  # minimum pixels movement to count as valid
    SHOW_VIDEO = False  # Set to False to process without display, True to show video
    
    frame_idx = 0
    print(f"Processing {total_frames} frames at {fps} fps...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            minute = frame_idx // (int(fps) * 60)
            
            # Get detections with tracking
            results = model.track(frame, persist=True, classes=[2, 7], conf=0.25)
            
            detected_tracks = set()
            
            # Draw exit lines on frame
            for line in exit_lines.values():
                coords = list(line.coords)
                for i in range(len(coords) - 1):
                    pt1 = tuple(map(int, coords[i]))
                    pt2 = tuple(map(int, coords[i + 1]))
                    cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
            
            # Process detections
            for result in results:
                for box in result.boxes:
                    if box.id is None:
                        continue
                    
                    track_id = int(box.id)
                    detected_tracks.add(track_id)
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    
                    # Use bottom-center point for more stable crossing detection
                    center_x = (x1 + x2) // 2
                    bottom_y = y2  # Bottom of bounding box
                    detection_point = Point(center_x, bottom_y)
                    
                    # Check if in ROI and not in exclusion zone
                    if roi_polygon.contains(detection_point) and (exclusion_polygon is None or not exclusion_polygon.contains(detection_point)):
                        
                        # Initialize track state if needed
                        if track_id not in track_state:
                            track_state[track_id] = {
                                'detected_in_roi': True,
                                'exits_crossed': {exit_id: False for exit_id in exit_lines.keys()},
                                'frame_count': 1,
                                'positions': [(center_x, bottom_y)]
                            }
                        else:
                            track_state[track_id]['frame_count'] += 1
                            track_state[track_id]['positions'].append((center_x, bottom_y))
                        
                        # Remove from inactive tracking
                        if track_id in inactive_tracks:
                            del inactive_tracks[track_id]
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Check exit line crossings (only count if not already counted)
                        for exit_id, line in exit_lines.items():
                            if not track_state[track_id]['exits_crossed'][exit_id]:
                                distance = line.distance(detection_point)
                                if distance < 10:  # Threshold for crossing
                                    # Validate track quality before counting
                                    if track_state[track_id]['frame_count'] >= MIN_TRACK_LENGTH:
                                        # Check minimum movement
                                        positions = track_state[track_id]['positions']
                                        if len(positions) > 1:
                                            movement = ((positions[-1][0] - positions[0][0])**2 + (positions[-1][1] - positions[0][1])**2)**0.5
                                            if movement >= MIN_MOVEMENT:
                                                # Count this crossing
                                                track_state[track_id]['exits_crossed'][exit_id] = True
                                                crossing_counts[minute][exit_id] += 1
                                                print(f"Frame {frame_idx}: Track {track_id} crossed {exit_id} (minute {minute})")
            
            # Track inactive vehicles
            for track_id in list(track_state.keys()):
                if track_id not in detected_tracks:
                    if track_id not in inactive_tracks:
                        inactive_tracks[track_id] = 1
                    else:
                        inactive_tracks[track_id] += 1
                    
                    # Remove if inactive too long
                    if inactive_tracks[track_id] > TRACK_TIMEOUT:
                        del track_state[track_id]
                        del inactive_tracks[track_id]
            
            # Display progress
            if frame_idx % 30 == 0:
                pct = (frame_idx / total_frames) * 100
                print(f"  Progress: {frame_idx}/{total_frames} frames ({pct:.1f}%)")
            
            # Display frame (optional)
            if SHOW_VIDEO:
                cv2.imshow('Car Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    
    # Release resources
    cap.release()
    if SHOW_VIDEO:
        cv2.destroyAllWindows()
    
    # Write aggregated results to CSV
    if crossing_counts:
        max_minute = max(crossing_counts.keys())
        all_exits = set()
        for minute_data in crossing_counts.values():
            all_exits.update(minute_data.keys())
        
        all_exits = sorted(list(all_exits))
        
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ['minute'] + all_exits
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for minute in range(int(max_minute) + 1):
                row = {'minute': minute}
                for exit_id in all_exits:
                    row[exit_id] = crossing_counts[minute].get(exit_id, 0)
                writer.writerow(row)
        
        print(f"\nAggregated results saved to {output_csv}")
    else:
        print("No crossings detected.")



if __name__ == "__main__":
    video_path = "data/roundabout.avi"
    model_path = "yolov8m.pt"
    roi_path = "output/roi.json"
    exit_lines_path = "output/exit_lines.json"
    output_csv = "output/car_crossings.csv"

    detect_cars(video_path, model_path, roi_path, exit_lines_path, output_csv)