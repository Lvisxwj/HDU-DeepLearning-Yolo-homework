# 🚗 Vehicle Submersion Detection System

**水积识别和车辆淹没部位判别系统**

A comprehensive system for detecting vehicles in flood images and analyzing their submersion levels using custom-trained YOLO models.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![YOLO](https://img.shields.io/badge/YOLO-v8%20%7C%20v11-00FFFF.svg)](https://github.com/ultralytics/ultralytics)

---

## 📋 Overview

This system uses custom-trained YOLO models to:
- **Detect vehicle parts** (body, window, wheel) in flood images
- **Segment water areas** using semantic segmentation
- **Calculate submersion levels** by analyzing part-water overlap
- **Generate statistics** and color-coded visualizations

Built for academic research on flood disaster assessment and vehicle damage analysis.

---

## ✨ Features

### 🎯 Part-Based Vehicle Detection
- Detects vehicle components separately: **body**, **window**, **wheel**
- Custom YOLO model trained on 100 epochs (exp23)
- Confidence threshold: 10% (adjustable 1-100%)

### 🌊 Water Segmentation
- Semantic segmentation for water area identification
- Custom YOLO segmentation model
- 60% opacity bright blue overlay for clear visualization

### 📊 Submersion Analysis
- **Fully Submerged**: ≥70% of body/window underwater
- **Partially Submerged**: Wheels underwater or 20-70% body coverage
- **Not Submerged**: <20% water coverage
- Per-part submersion ratio calculation

### 🎨 Interactive Web Interface
- Built with Streamlit for easy interaction
- Multi-model selection (custom + official YOLO models)
- Real-time processing with progress indicators
- Three view modes: Detection Only, Segmentation Only, Combined
- Split-screen comparison (original vs processed)

### 📥 Export & Download
- Download all results as ZIP archive
- Individual downloads per image (detection, segmentation, combined, stats JSON)
- Aggregate statistics across multiple images
- CSV export support

---

## 🚀 Quick Start

### Prerequisites

```bash
# Conda environment with Python 3.8+
conda activate yolov11

# Required packages
pip install -r requirements.txt
```

### Run the Application

```bash
# Method 1: Direct command
streamlit run app.py

# Method 2: Batch file (Windows)
run.bat
```

The web interface will open at `http://localhost:8501`

---

## 📁 Project Structure

```
Project3/
├── README.md                   # This file
├── GUIDE.md                    # Quick start guide
├── run.bat                     # Application launcher
├── requirements.txt            # Python dependencies
│
├── Core Application
│   ├── app.py                  # Streamlit web interface
│   ├── detection.py            # Vehicle part detection module
│   ├── segmentation.py         # Water segmentation module
│   ├── utils.py                # Analysis and visualization
│   └── config.py               # Configuration and model paths
│
├── models/                     # Trained YOLO models
│   ├── vehical/
│   │   ├── best_exp23.pt      # Best vehicle detection (12.37% max conf)
│   │   ├── last_exp23.pt      # Last epoch checkpoint
│   │   ├── best.pt            # Improved model (0.44% max conf)
│   │   └── last.pt
│   └── flood/
│       ├── best.pt            # Best water segmentation
│       └── last.pt            # Last epoch checkpoint
│
├── demo_images/                # Sample flood images
├── results/                    # Processing outputs
├── uploads/                    # User uploaded images
│
├── README/                     # Documentation
│   ├── README.md              # Complete documentation
│   ├── requirement.md         # Project requirements
│   ├── project_details.md     # Implementation details
│   ├── architecture.md        # System architecture
│   ├── CUSTOM_MODELS.md       # Custom model info
│   ├── CUSTOM_MODEL_CLASSES.md # Class definitions
│   └── SETUP_COMPLETE.md      # Setup guide
│
└── test/                       # Test scripts (19 files)
    ├── test_flood_now.bat/.py
    ├── test_exp23.bat/.py
    ├── test_custom.bat/.py
    └── ...
```

---

## 🎯 Usage Guide

### 1. Launch Application
```bash
streamlit run app.py
```

### 2. Configure Models (Sidebar)
- **Detection Model**: Select vehicle detection model
  - Recommended: `🎯 Vehicle Custom (Exp23 Best)`
- **Segmentation Model**: Select water segmentation model
  - Recommended: `🌊 Flood Custom (Best)`
- **Confidence Threshold**: Adjust sensitivity (default: 0.10)

### 3. Process Images (Tab 1)
1. Click "Browse files" to upload images
2. Select one or multiple images (JPG, PNG, BMP)
3. Click "🚀 Process Images"
4. View results with radio buttons:
   - **Detection Only**: Vehicle bounding boxes only
   - **Segmentation Only**: Water segmentation only
   - **Combined**: Both detection and segmentation

### 4. View Statistics (Tab 2)
- Overall statistics across all images
- Per-image breakdown table
- Submersion level distribution chart

### 5. Download Results (Tab 3)
- Download all results as ZIP
- Or download individual results per image

---

## 🧪 Testing

Test scripts are located in the `test/` directory:

```bash
# Test flood segmentation
cd test
test_flood_now.bat

# Test exp23 vehicle detection
test_exp23.bat

# Test all images in demo_images/
test_all.bat

# Quick single-image test
quick_test.bat
```

---

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Model paths
DETECTION_MODELS = {
    "🎯 Vehicle Custom (Exp23 Best)": "models/vehical/best_exp23.pt",
    # Add more models...
}

SEGMENTATION_MODELS = {
    "🌊 Flood Custom (Best)": "models/flood/best.pt",
    # Add more models...
}

# Custom class definitions
CUSTOM_VEHICLE_PARTS = {0: 'body', 1: 'window', 2: 'wheel'}
CUSTOM_WATER_CLASSES = {0: 'water'}
```

---

## 📊 Technical Details

### Models

#### Vehicle Detection Model (Exp23)
- **Architecture**: YOLOv8/v11 object detection
- **Classes**: body (0), window (1), wheel (2)
- **Training**: 100 epochs on project1 dataset
- **Performance**: 12.37% max confidence on test set
- **Input**: Ground-level vehicle images

#### Water Segmentation Model
- **Architecture**: YOLOv8/v11 semantic segmentation
- **Classes**: water (0)
- **Training**: Custom flood dataset
- **Output**: Binary mask for water regions

### Submersion Analysis Algorithm

```python
# Whole vehicle submersion
if submersion_ratio >= 0.7:
    level = "fully_submerged"
elif submersion_ratio >= 0.2:
    level = "partially_submerged"
else:
    level = "not_submerged"

# Part-based submersion (custom model)
if part == 'wheel' and submersion_ratio >= 0.5:
    level = "partially_submerged"
elif part in ['body', 'window'] and submersion_ratio >= 0.7:
    level = "fully_submerged"
```

### Visualization
- **Red**: Fully Submerged (≥70%)
- **Orange**: Partially Submerged (20-70%)
- **Green**: Not Submerged (<20%)
- **Blue**: Water Area (60% opacity overlay)

---

## ⚠️ Important Notes

### Model Limitations
The custom vehicle detection model was trained on **ground-level clear vehicle photos**, NOT aerial flood images. For optimal results:

✅ **Best**: Use images from training dataset
- Location: `机器学习课程实践\project1\dataset2\images\test\`

⚠️ **Limited**: New aerial flood images (cc*.jpg)
- Detection may be poor due to angle/lighting differences
- Consider retraining with flood-specific images

🔄 **Alternative**: Use official YOLO models (YOLOv11n) for general vehicles

### Confidence Threshold
- **Default**: 0.10 (10%)
- **Range**: 0.01 to 1.0
- Lower = more detections (more false positives)
- Higher = fewer detections (higher precision)

---

## 🐛 Troubleshooting

### No Detections?
1. Lower confidence threshold (try 0.05)
2. Use images from training dataset
3. Check console output for model loading confirmation

### Water Not Visible?
1. Water overlay is 60% opacity with bright blue
2. Check console for segmentation results
3. Verify water exists in image

### Results Disappear When Switching Views?
- **Fixed**: Results now persist across view switches
- Ensure you're in "Upload & Process" tab

---

## 📚 Documentation

Detailed documentation available in `README/`:

- **[README.md](README/README.md)**: Complete system documentation
- **[GUIDE.md](GUIDE.md)**: Quick start guide
- **[requirement.md](README/requirement.md)**: Original requirements
- **[architecture.md](README/architecture.md)**: Technical architecture
- **[CUSTOM_MODELS.md](README/CUSTOM_MODELS.md)**: Model information
- **[project_details.md](README/project_details.md)**: Implementation notes

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit 1.28+
- **Backend**: Python 3.8+
- **Deep Learning**: PyTorch 2.0+ with CUDA
- **Computer Vision**: OpenCV 4.8+
- **Object Detection**: Ultralytics YOLO (v8 & v11)
- **Data Processing**: NumPy, Pandas
- **Image Handling**: PIL/Pillow

---

## 📦 Dependencies

```txt
streamlit>=1.28.0
ultralytics>=8.0.0
opencv-python>=4.8.0
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🎓 Academic Context

**Project**: Task 4 - Vehicle Submersion Detection System
**Course**: Machine Learning Practice (机器学习课程实践)
**Institution**: [Your Institution]
**Due Date**: October 28, 2025
**Environment**: Conda (yolov11), Python 3.x

---

## 📝 License

This project is for academic use. Please refer to your institution's policies for code sharing and reuse.

---

## 🤝 Contributing

This is an academic project. For questions or issues:
1. Check `README/` documentation
2. Review `test/` scripts for examples
3. Consult `GUIDE.md` for troubleshooting

---

## 📧 Contact

For questions related to this project, please contact the development team.

---

## 🙏 Acknowledgments

- **Ultralytics YOLO**: Object detection framework
- **Streamlit**: Web interface framework
- **OpenCV**: Image processing library
- **PyTorch**: Deep learning platform

---

## 📊 Project Statistics

- **Lines of Code**: ~2,000+
- **Python Files**: 5 core modules
- **Test Scripts**: 19 test files
- **Documentation**: 7 markdown files
- **Models**: 6 trained models (2 best + 4 checkpoints)

---

**⭐ Star this repository if you find it helpful!**

**🚀 Get started:** `streamlit run app.py`
