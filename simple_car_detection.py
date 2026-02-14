import cv2
from ultralytics import YOLO
import torch
import json
from shapely.geometry import Point, Polygon, LineString
import csv
import torch
print("CUDA available:", torch.cuda.is_available())

def detect_cars(video_path, model_path, roi_path, exit_lines_path, output_csv):
    # Load YOLO model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(model_path)
    model.to(device)  # Explicitly move the model to GPU if available

    # Load ROI polygon
    with open(roi_path, 'r') as f:
        roi_data = json.load(f)
    roi_polygon = Polygon(roi_data[0])

    # Load exit lines
    with open(exit_lines_path, 'r') as f:
        exit_lines_data = json.load(f)
    exit_lines = {key: LineString(value) if len(value) == 2 else LineString(value) for key, value in exit_lines_data.items()}

    # Initialize CSV logging
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['frame'] + list(exit_lines.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            return

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            row = {'frame': frame_count}
            for line_name in exit_lines.keys():
                row[line_name] = 0

            # Draw exit lines on the frame
            for line in exit_lines.values():
                pt1, pt2 = tuple(map(int, line.coords[0])), tuple(map(int, line.coords[1]))
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

            # Perform detection with tracking enabled
            results = model.track(frame, persist=True, classes=[2, 7], conf=0.25)  # Enable BYTETrack for cars and trucks

            # Draw bounding boxes and labels on the frame
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    label = f"Car {conf:.2f}"

                    # Check if the center of the bounding box is inside the ROI
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    center_point = Point(center_x, center_y)

                    if roi_polygon.contains(center_point):
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Check if the car crosses any exit lines
                        for line_name, line in exit_lines.items():
                            if line.distance(center_point) < 5:  # Threshold for crossing detection
                                print(f"Car crossed {line_name}")
                                row[line_name] += 1

            # Write row to CSV
            writer.writerow(row)

            # Display the frame
            cv2.imshow('Car Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "data/roundabout.avi"
    model_path = "yolov8n.pt"
    roi_path = "output/roi.json"
    exit_lines_path = "output/exit_lines.json"
    output_csv = "output/car_crossings.csv"

    detect_cars(video_path, model_path, roi_path, exit_lines_path, output_csv)