import cv2
from ultralytics import YOLO
import torch
import json
from shapely.geometry import Point, Polygon, LineString
import csv
import os
import numpy as np
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

    # Load exit lines/polygons
    with open(exit_lines_path, 'r') as f:
        exit_lines_data = json.load(f)
    exit_lines = {key: Polygon(value) for key, value in exit_lines_data.items()}

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
    # track_id -> {
    #   'detected_in_roi': bool,
    #   'exits_state': {exit_id: {'inside': bool, 'counted': bool}},
    #   'frame_count': int,
    #   'last_position': (x, y)
    # }
    track_state = {}
    crossing_counts = defaultdict(lambda: defaultdict(int))  # minute -> exit_id -> count
    inactive_tracks = {}  # track_id -> frames_inactive
    
    TRACK_TIMEOUT = 30  # frames
    MIN_TRACK_LENGTH = 5  # minimum frames to be considered valid
    MIN_MOVEMENT = 10  # minimum pixels movement to count as valid
    SHOW_VIDEO = True  # Set to False to process without display, True to show video
    
    frame_idx = 0
    print(f"Processing {total_frames} frames at {fps} fps...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            minute = frame_idx // (int(fps) * 60)
            
            # Get detections with tracking (class 2 = car, class 7 = truck)
            results = model.track(frame, persist=True, classes=[2, 7], conf=0.25)
            
            detected_tracks = set()
            
            # Draw exit polygons on frame
            for exit_id, polygon in exit_lines.items():
                coords = list(polygon.exterior.coords)
                pts = np.array(coords, dtype=np.int32)
                cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
                
                # Add counter for this exit
                exit_count = crossing_counts[minute].get(exit_id, 0)
                # Find centroid of polygon for text placement
                centroid_x = int(np.mean([c[0] for c in coords]))
                centroid_y = int(np.mean([c[1] for c in coords]))
                cv2.putText(frame, f"{exit_id}: {exit_count}", (centroid_x - 30, centroid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
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
                                'exits_state': {exit_id: {'inside': False, 'counted': False} for exit_id in exit_lines.keys()},
                                'frame_count': 1,
                                'last_position': (center_x, bottom_y)
                            }
                        else:
                            track_state[track_id]['frame_count'] += 1
                            track_state[track_id]['last_position'] = (center_x, bottom_y)
                        
                        # Remove from inactive tracking
                        if track_id in inactive_tracks:
                            del inactive_tracks[track_id]
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Check exit polygon state transitions (state-based counting)
                        for exit_id, polygon in exit_lines.items():
                            current_inside = polygon.contains(detection_point)
                            previous_inside = track_state[track_id]['exits_state'][exit_id]['inside']
                            counted = track_state[track_id]['exits_state'][exit_id]['counted']
                            
                            # Update current state
                            track_state[track_id]['exits_state'][exit_id]['inside'] = current_inside
                            
                            # Count only on transition: was inside, now outside, and not yet counted
                            if previous_inside and not current_inside and not counted:
                                # Validate track quality before counting
                                if track_state[track_id]['frame_count'] >= MIN_TRACK_LENGTH:
                                    # Mark as counted to prevent duplicate counts
                                    track_state[track_id]['exits_state'][exit_id]['counted'] = True
                                    crossing_counts[minute][exit_id] += 1
                                    print(f"Frame {frame_idx}: Track {track_id} exited {exit_id} (minute {minute})")
            
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
            
            # Add global counter display in top-left corner
            y_offset = 30
            cv2.putText(frame, f"Minute: {minute}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (10, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Tracked: {len(track_state)}", (10, y_offset + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
            
            # Add statistics in bottom-right corner (two tables side by side)
            frame_height, frame_width = frame.shape[:2]
            
            # Calculate cumulative totals from start
            cumulative_totals = {exit_id: 0 for exit_id in exit_lines.keys()}
            for m in range(minute + 1):
                for exit_id in exit_lines.keys():
                    cumulative_totals[exit_id] += crossing_counts[m].get(exit_id, 0)
            
            # Layout: two columns side by side
            box_height = 60 + len(exit_lines) * 22
            col_width = 150
            total_box_width = col_width * 2 + 20  # Two columns + gap
            
            box_x = frame_width - total_box_width - 10
            box_y = frame_height - box_height - 10
            
            # Draw white background rectangle for both columns
            cv2.rectangle(frame, (box_x, box_y), (frame_width - 10, frame_height - 10), (255, 255, 255), -1)
            cv2.rectangle(frame, (box_x, box_y), (frame_width - 10, frame_height - 10), (0, 0, 0), 2)
            
            # LEFT COLUMN: Exits this minute
            text_y = box_y + 25
            cv2.putText(frame, "This minute:", (box_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            text_y += 22
            
            for exit_id in sorted(exit_lines.keys()):
                exit_count = crossing_counts[minute].get(exit_id, 0)
                cv2.putText(frame, f"{exit_id}: {exit_count}", (box_x + 15, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)
                text_y += 20
            
            # RIGHT COLUMN: Total since start
            text_y = box_y + 25
            cv2.putText(frame, "Total:", (box_x + col_width + 15, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
            text_y += 22
            
            for exit_id in sorted(exit_lines.keys()):
                total_count = cumulative_totals[exit_id]
                cv2.putText(frame, f"{exit_id}: {total_count}", (box_x + col_width + 20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 200), 1)
                text_y += 20
            
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
        
        # Calculate totals for each exit
        exit_totals = {exit_id: 0 for exit_id in all_exits}
        for minute_data in crossing_counts.values():
            for exit_id in all_exits:
                exit_totals[exit_id] += minute_data.get(exit_id, 0)
        
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ['minute'] + all_exits
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write data for each minute
            for minute in range(int(max_minute) + 1):
                row = {'minute': minute}
                for exit_id in all_exits:
                    row[exit_id] = crossing_counts[minute].get(exit_id, 0)
                writer.writerow(row)
            
            # Write total row
            total_row = {'minute': 'TOTAL'}
            for exit_id in all_exits:
                total_row[exit_id] = exit_totals[exit_id]
            writer.writerow(total_row)
        
        print(f"\nAggregated results saved to {output_csv}")
        print(f"Total exits: {exit_totals}")
    else:
        print("No crossings detected.")



if __name__ == "__main__":
    video_path = "data/roundabout.avi"
    model_path = "yolov8m.pt"
    roi_path = "output/roi.json"
    exit_lines_path = "output/exit_lines.json"
    output_csv = "output/car_crossings.csv"

    detect_cars(video_path, model_path, roi_path, exit_lines_path, output_csv)