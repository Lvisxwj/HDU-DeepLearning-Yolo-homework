# 水积识别和车辆淹没部位判别系统
# Vehicle Submersion Detection System

A comprehensive system for detecting vehicles in flooded areas and analyzing their submersion levels using YOLO models.

## Features

### ✅ Subtask 4.1: Image Upload and Download
- **Upload Images**: Support for multiple image formats (JPG, PNG, BMP)
- **Download Results**: Export detection and segmentation results

### ✅ Subtask 4.2: Split-Screen Display
- Side-by-side comparison of original images and processed results
- Multiple view options: Detection only, Segmentation only, Combined

### ✅ Subtask 4.3: Vehicle Statistics
- Count vehicles by submersion level:
  - Fully Submerged (>70% underwater)
  - Partially Submerged (20-70% underwater)
  - Not Submerged (<20% underwater)
- Detailed per-vehicle analysis
- Aggregate statistics across all images

### ✅ Subtask 4.4: Model Selection
- **🎯 Custom Trained Models** (Your Models):
  - Vehicle Custom (Best) - Optimized vehicle detection
  - Vehicle Custom (Last) - Alternative vehicle detection
  - Flood Custom (Best) - Optimized flood/water segmentation
  - Flood Custom (Last) - Alternative flood segmentation
- **Object Detection Models** (Official YOLO):
  - YOLOv11n, YOLOv11s, YOLOv11m, YOLOv11l
  - YOLOv8n, YOLOv8s
- **Semantic Segmentation Models** (Official YOLO):
  - YOLOv11n-seg, YOLOv11s-seg, YOLOv11m-seg
  - YOLOv8n-seg, YOLOv8s-seg

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended)
- Conda environment with ultralytics installed

### Setup

1. **Activate your conda environment:**
```bash
conda activate yolov11
```

2. **Install additional dependencies (if needed):**
```bash
pip install -r requirements.txt
```

Note: Since you already have ultralytics and CUDA configured, you may only need to install Streamlit and other web dependencies.

3. **Test your custom models (recommended):**
```bash
python test_custom_models.py
```

## 🎯 Custom Models

This system includes **your custom trained models**:

- **Vehicle Detection**: `models/vehical/best.pt` and `last.pt`
- **Flood Segmentation**: `models/flood/best.pt` and `last.pt`

These models appear at the top of the dropdown menus marked with 🎯 and 🌊 icons.

**Recommended for your demo:**
- Detection Model: `🎯 Vehicle Custom (Best)`
- Segmentation Model: `🌊 Flood Custom (Best)`

See [CUSTOM_MODELS.md](CUSTOM_MODELS.md) for detailed information.

## Usage

### Running the Application

1. **Start the Streamlit application:**
```bash
streamlit run app.py
```

Or use the batch file:
```bash
run.bat
```

2. **Open your web browser:**
The application will automatically open at `http://localhost:8501`

### Using the System

1. **Select Models** (Left Sidebar):
   - Choose your preferred object detection model
   - Choose your preferred segmentation model
   - Adjust confidence threshold if needed

2. **Upload Images** (Upload & Process Tab):
   - Click "Browse files" to select images
   - Upload single or multiple images
   - Click "Process Images" to start analysis

3. **View Results**:
   - Original image displayed on the left
   - Processed results on the right
   - Switch between Detection, Segmentation, and Combined views
   - View statistics for each image

4. **Check Statistics** (Statistics Tab):
   - Overall statistics across all processed images
   - Bar chart visualization
   - Per-image breakdown table

5. **Download Results** (Download Results Tab):
   - Download all results as a ZIP file
   - Download individual results per image
   - Export formats: PNG (images), JSON (statistics)

## Project Structure

```
Project3/
├── app.py                    # Main Streamlit application
├── detection.py             # Vehicle detection module (YOLO)
├── segmentation.py          # Water segmentation module (YOLO-seg)
├── utils.py                 # Utility functions for statistics and visualization
├── config.py                # Configuration (models, thresholds, etc.)
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── CUSTOM_MODELS.md        # Custom model documentation
├── architecture.md         # System architecture documentation
├── run.bat                 # Windows run script
├── test_system.py          # System verification script
├── test_custom_models.py   # Custom model testing script
├── uploads/                # Uploaded images storage
├── results/                # Processed results storage
├── models/                 # Model files
│   ├── vehical/           # Custom vehicle detection models
│   │   ├── best.pt        # Best checkpoint (recommended)
│   │   └── last.pt        # Last checkpoint
│   └── flood/             # Custom flood segmentation models
│       ├── best.pt        # Best checkpoint (recommended)
│       └── last.pt        # Last checkpoint
└── temp/                   # Temporary files
```

## Technical Details

### Detection Module (`detection.py`)
- Uses YOLO for vehicle detection
- Supports multiple YOLO versions
- Detects: cars, motorcycles, buses, trucks
- Configurable confidence threshold

### Segmentation Module (`segmentation.py`)
- Uses YOLO segmentation models
- Identifies water/flood regions
- Provides pixel-level masks
- Model switching capability

### Statistics Module (`utils.py`)
- Calculates vehicle submersion levels
- Analyzes which parts are submerged (top, middle, bottom, wheels)
- Color-coded visualization:
  - 🔴 Red: Fully Submerged
  - 🟠 Orange: Partially Submerged
  - 🟢 Green: Not Submerged
  - 🔵 Blue: Water Area

## Models

The system will automatically download required YOLO models on first use:
- Detection models: ~3-50 MB each
- Segmentation models: ~3-50 MB each

Models are cached locally in the `models/` directory.

## Performance Tips

1. **For faster processing**: Use smaller models (YOLOv11n, YOLOv8n)
2. **For better accuracy**: Use larger models (YOLOv11l, YOLOv11m)
3. **GPU acceleration**: Ensure CUDA is properly configured
4. **Batch processing**: Upload multiple images at once

## Troubleshooting

### Models not downloading
- Check internet connection
- Ensure sufficient disk space
- Try manually downloading from Ultralytics

### Slow performance
- Use GPU if available
- Reduce image size
- Use smaller models
- Lower confidence threshold

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Activate the correct conda environment: `conda activate yolov11`

## Requirements

- Python >= 3.8
- ultralytics >= 8.0.0
- streamlit >= 1.28.0
- opencv-python >= 4.8.0
- torch >= 2.0.0 (with CUDA support)
- numpy >= 1.24.0
- Pillow >= 10.0.0
- pandas >= 2.0.0

## Acceptance Criteria

✅ **Subtask 4.1 (2/2 points)**:
- Image upload supported
- Detection and segmentation results download supported

✅ **Subtask 4.2 (1 point)**:
- Split-screen display implemented

✅ **Subtask 4.3 (1 point)**:
- Vehicle counting by submersion level implemented

✅ **Subtask 4.4 (2/2 points)**:
- Multiple detection models selectable
- Multiple segmentation models selectable

## Deadline

**October 28, 2025** - On-site system demonstration required

## Author

Built for Project 3 - Water Accumulation Identification and Vehicle Submersion Detection System

## License

Educational use only
