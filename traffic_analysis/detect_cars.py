import cv2
import os
import sys
sys.path.insert(0, 'C:/Users/Metr/Documents/GitHub/cdv-nabor-mikeska/traffic_analysis/yolov5')  # Adjust path to YOLOv5 repo
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator
from utils.torch_utils import select_device

def detect_cars(video_path, model_path, output_dir):
    # Load YOLOv5 model
    model = DetectMultiBackend(model_path, device=select_device(''))

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Prepare output video writer
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'output.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)
        
        # Draw bounding boxes and labels on the frame
        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(frame)

        # Display the frame (optional)
        cv2.imshow('Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "C:\\Users\\Metr\\Documents\\GitHub\\cdv-nabor-mikeska\\traffic_analysis\\data\\roundabout.avi"
    model_path = "C:\\Users\\Metr\\Documents\\GitHub\\cdv-nabor-mikeska\\traffic_analysis\\yolov8n.pt"
    output_dir = "C:\\Users\\Metr\\Documents\\GitHub\\cdv-nabor-mikeska\\traffic_analysis\\output"

    detect_cars(video_path, model_path, output_dir)