# Vehicle Detection and Counting at Roundabout

This project provides Python scripts for analyzing video recordings from a roundabout. It uses YOLOv8 object detection to identify vehicles (cars and trucks) and counts how many vehicles exit the roundabout through each exit.

## Project Structure

```
.
├── data/
│   └── roundabout.avi              (input video file)
├── output/
│   ├── roi.json                    (ROI polygon configuration)
│   ├── exit_lines.json             (exit polygon definitions)
│   ├── exclusion.json              (exclusion zone polygon)
│   └── car_crossings.csv           (output results)
├── 01_detect_and_count.py          (main detection and counting script)
├── yolov8m.pt                      (YOLOv8 model file)
├── requirements.txt                (project dependencies)
└── README.md
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd cdv-nabor-mikeska
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   ```
   
   Activate environment:
   - Windows (PowerShell):
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv8 model:**
   The script requires `yolov8m.pt` model file. Place it in the project root directory.

## Configuration

The setup process requires two configuration steps:

### Step 1: Define Region of Interest (ROI)
Run the ROI setup script to define the area where vehicles should be detected:
```bash
python 00_setup_detection_roi.py
```
This creates `output/roi.json` with the polygon coordinates of the detection area.

### Step 2: Define Exit Polygons
Run the exit polygon setup script to define each exit area:
```bash
python 00_setup_exit_polygons.py
```
This creates `output/exit_lines.json` with polygon definitions for each exit.

**Optional:** You can also create an exclusion zone by creating `output/exclusion.json` with polygon coordinates for areas to exclude from detection.

## Usage

1. **Place input video:**
   Ensure your video file (`roundabout.avi`) is placed in the `data/` directory.

2. **Run detection and counting:**
   ```bash
   python 01_detect_and_count.py
   ```

3. **View results:**
   Results are saved to `output/car_crossings.csv` with the following structure:
   - First column: minute number
   - Subsequent columns: exit IDs with vehicle counts
   - Last row: total counts for each exit

## Output Format

The output CSV file contains:
- **Minute-by-minute breakdown** of vehicle counts for each exit
- **TOTAL row** at the end with cumulative counts

Example:
```
minute,exit_1,exit_2,exit_3
0,5,3,2
1,4,6,1
2,3,4,5
TOTAL,12,13,8
```

## Features

- **YOLOv8 Detection:** Uses YOLOv8 medium model for real-time vehicle detection
- **Multi-class detection:** Detects both cars (class 2) and trucks (class 7)
- **Polygon-based ROI:** Flexible region of interest using geometric polygons
- **State-based counting:** Prevents duplicate counting using state transitions
- **GPU support:** Automatically uses CUDA if available
- **Progress tracking:** Shows real-time progress and frame statistics
- **Exclusion zones:** Support for excluding specific areas from detection

## Configuration Parameters

Key parameters in `01_detect_and_count.py`:
- `TRACK_TIMEOUT`: Frames to wait before removing inactive tracks (default: 30)
- `MIN_TRACK_LENGTH`: Minimum frames required to count a vehicle (default: 5)
- `SHOW_VIDEO`: Set to `True` to display video with detections (default: False)
- Confidence threshold: 0.30 (adjustable in model.track())

## Notes

- The detection uses a confidence threshold of 0.30 for balancing accuracy and recall
- Video output is disabled by default for faster processing (set `SHOW_VIDEO = True` to enable)
- Processing speed depends on video quality and GPU availability
- All console output from YOLO is suppressed for cleaner logs

## System Requirements

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for faster processing)
- Sufficient disk space for model file (~50MB for YOLOv8m)
