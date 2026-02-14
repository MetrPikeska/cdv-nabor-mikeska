import cv2
from ultralytics import YOLO
import torch
import numpy as np
import csv
import json

# Configuration
DEBUG = False  # Set to True to enable visualization
PROCESS_EVERY_N_FRAMES = 5  # Process every N-th frame
LOW_RESOLUTION = (640, 360)  # Low-resolution stream dimensions
FPS = 25  # Original video FPS

# Load ROI mask from JSON
ROI_JSON_PATH = "output/roi.json"
with open(ROI_JSON_PATH, 'r') as f:
    roi_data = json.load(f)
roi_mask = np.zeros(LOW_RESOLUTION, dtype=np.uint8)
cv2.fillPoly(roi_mask, [np.array(roi_data[0], dtype=np.int32)], 255)

# Define ROI lines for exits
EXIT_LINES = {
    "exit_1": [(100, 300), (200, 300)],
    "exit_2": [(300, 100), (400, 100)],
}

# Initialize vehicle counts and tracking
vehicle_counts = {exit_name: 0 for exit_name in EXIT_LINES}
tracked_vehicles = {exit_name: set() for exit_name in EXIT_LINES}

# Helper function to check if a point crosses a line
def crosses_line(centroid, line, prev_positions):
    if len(prev_positions) < 2:
        return False
    prev_x, prev_y = prev_positions[-2]
    curr_x, curr_y = centroid
    line_start, line_end = line

    # Check if the line is crossed between the previous and current positions
    return cv2.clipLine((prev_x, prev_y, curr_x, curr_y), line_start, line_end)

# Main detection function
def detect_cars(video_path, model_path):
    # Load YOLO model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(model_path)
    model.to(device)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # Set video resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, LOW_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, LOW_RESOLUTION[1])

    frame_index = 0
    results_per_minute = {}
    vehicle_positions = {}  # Track vehicle positions by ID

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to reduce computational load
        if frame_index % PROCESS_EVERY_N_FRAMES == 0:
            # Perform detection
            results = model(frame, classes=[2, 7], conf=0.25)

            # Process detections
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

                    # Check if centroid is within the bounds of the ROI mask
                    if 0 <= centroid[1] < roi_mask.shape[0] and 0 <= centroid[0] < roi_mask.shape[1]:
                        if roi_mask[centroid[1], centroid[0]] > 0:
                            # Track vehicle positions
                            vehicle_id = f"{centroid[0]}_{centroid[1]}"
                            if vehicle_id not in vehicle_positions:
                                vehicle_positions[vehicle_id] = []
                            vehicle_positions[vehicle_id].append(centroid)

                            # Check if the vehicle crosses any exit line
                            for exit_name, line in EXIT_LINES.items():
                                if crosses_line(centroid, line, vehicle_positions[vehicle_id]):
                                    if vehicle_id not in tracked_vehicles[exit_name]:
                                        tracked_vehicles[exit_name].add(vehicle_id)
                                        vehicle_counts[exit_name] += 1

                            if DEBUG:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.circle(frame, centroid, 5, (0, 0, 255), -1)

        # Update frame index
        frame_index += 1

        # Display frame if DEBUG is enabled
        if DEBUG:
            cv2.imshow('Car Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Aggregate results per minute
        minute = frame_index // (FPS * 60)
        if minute not in results_per_minute:
            results_per_minute[minute] = {exit_name: 0 for exit_name in EXIT_LINES}
        for exit_name in EXIT_LINES:
            results_per_minute[minute][exit_name] = vehicle_counts[exit_name]

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Export results to CSV
    with open('output/vehicle_counts.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Minute"] + list(EXIT_LINES.keys()))
        for minute, counts in results_per_minute.items():
            writer.writerow([minute] + [counts[exit_name] for exit_name in EXIT_LINES])

    print("Vehicle counts exported to output/vehicle_counts.csv")

if __name__ == "__main__":
    video_path = "data/roundabout.avi"
    model_path = "yolov8n.pt"

    detect_cars(video_path, model_path)