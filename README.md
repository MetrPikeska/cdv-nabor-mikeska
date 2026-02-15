# Vehicle Detection and Counting at Roundabout

This project provides Python scripts for analyzing video recordings from a roundabout. It uses YOLOv8 object detection to identify vehicles (cars and trucks) and counts how many vehicles exit the roundabout through each exit.

## Project Structure

```
.
├── data/
│   └── roundabout.avi              (input video file - add your own)
├── output/
│   ├── roi.json                    (ROI polygon configuration)
│   ├── exit_lines.json             (exit polygon definitions)
│   ├── exclusion.json              (exclusion zone polygon)
│   └── car_crossings.csv           (output results)
├── 00_setup_detection_roi.py       (script to define ROI)
├── 00_setup_exit_polygons.py       (script to define exit polygons)
├── 01_detect_and_count.py          (main detection and counting script)
├── yolov8m.pt                      (YOLOv8 model file - MUST be downloaded separately)
├── requirements.txt                (project dependencies)
└── README.md
```

**Note:** The ZIP file does NOT include:
- `yolov8m.pt` - Download separately (see Installation Step 4)
- `data/roundabout.avi` - Add your own video file
- `.venv/` - Create with `python -m venv .venv`

## Installation

### Step 0: Extract the ZIP File
The project is distributed as a compressed ZIP file. Extract it to get started:

1. **Windows (File Explorer):**
   - Right-click on `cdv-nabor-mikeska.zip`
   - Select "Extract All..."
   - Choose destination folder
   - Click "Extract"

2. **Command Line (Windows PowerShell):**
   ```powershell
   Expand-Archive -Path "cdv-nabor-mikeska.zip" -DestinationPath "C:\path\to\extract"
   cd C:\path\to\extract\cdv-nabor-mikeska
   ```

3. **macOS/Linux (Terminal):**
   ```bash
   unzip cdv-nabor-mikeska.zip
   cd cdv-nabor-mikeska
   ```

### Step 1: Clone the repository or Prepare the Project
If extracted from ZIP, the project is already ready. If cloning from Git:
```bash
git clone <repository_url>
cd cdv-nabor-mikeska
```

### Step 2: Create and Activate Virtual Environment
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

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download YOLOv8 Model
The YOLOv8 model file is **NOT included in the ZIP** due to its size (~50MB). You must download it separately:

**Option A: Automatic Download (Recommended)**
Simply run the detection script - it will auto-download the model on first run:
```bash
python 01_detect_and_count.py
```
The script will download `yolov8m.pt` automatically to the project root.

**Option B: Manual Download**
1. Download from Ultralytics releases:
   - Visit: https://github.com/ultralytics/assets/releases
   - Download `yolov8m.pt`
   
2. Place in project root directory:
   ```
   cdv-nabor-mikeska/
   ├── yolov8m.pt
   ├── 01_detect_and_count.py
   └── ...
   ```

3. Or download via command line:
   ```bash
   # Windows
   Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt" -OutFile "yolov8m.pt"
   
   # macOS/Linux
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
   ```

**Verify Installation:**
```bash
# Check if model is present
ls yolov8m.pt  # macOS/Linux
dir yolov8m.pt  # Windows
```

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
