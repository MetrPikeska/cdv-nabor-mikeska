import cv2
import torch
import os
from collections import defaultdict
from datetime import datetime, timedelta

def detect_and_count_vehicles(video_path, model_path, output_dir):
    # Load YOLOv8 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    # Initialize counters
    vehicle_counts = defaultdict(lambda: defaultdict(int))  # {minute: {branch: count}}

    # Define regions for branches (example coordinates, adjust as needed)
    branches = {
        "north": [(100, 50), (200, 150)],
        "east": [(300, 200), (400, 300)],
        "south": [(100, 400), (200, 500)],
        "west": [(50, 200), (150, 300)]
    }

    # Process video frame by frame
    current_time = 0  # in seconds
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)

        # Get current minute
        current_minute = int(current_time // 60)

        # Check detections
        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, box)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Determine which branch the vehicle is exiting
            for branch, ((bx1, by1), (bx2, by2)) in branches.items():
                if bx1 <= center_x <= bx2 and by1 <= center_y <= by2:
                    vehicle_counts[current_minute][branch] += 1

        # Update current time
        current_time += 1 / fps

    # Release resources
    cap.release()

    # Save results to a file
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'vehicle_counts.txt')
    with open(output_path, 'w') as f:
        f.write("Minute\tNorth\tEast\tSouth\tWest\n")
        for minute, counts in sorted(vehicle_counts.items()):
            f.write(f"{minute}\t{counts['north']}\t{counts['east']}\t{counts['south']}\t{counts['west']}\n")

    print(f"Vehicle counts saved to {output_path}")

if __name__ == "__main__":
    video_path = "C:\\Users\\Metr\\Documents\\GitHub\\cdv-nabor-mikeska\\traffic_analysis\\data\\roundabout.avi"
    model_path = "C:\\Users\\Metr\\Documents\\GitHub\\cdv-nabor-mikeska\\traffic_analysis\\yolov8n.pt"
    output_dir = "C:\\Users\\Metr\\Documents\\GitHub\\cdv-nabor-mikeska\\traffic_analysis\\output"

    detect_and_count_vehicles(video_path, model_path, output_dir)