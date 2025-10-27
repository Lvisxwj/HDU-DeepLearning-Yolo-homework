# 🚗 Vehicle Submersion Detection System - Quick Start Guide

**水积识别和车辆淹没部位判别系统 - 快速入门指南**

## 📋 Overview

This system uses custom-trained YOLO models to detect vehicle parts (body, window, wheel) and segment water areas in flood images, then calculates submersion levels for each detected part.

---

## 🚀 Quick Start

### 1. Launch the Application

```bash
# Activate environment
conda activate yolov11

# Run the web interface
streamlit run app.py
```

Or use the batch file:
```bash
run.bat
```

### 2. Use the System

1. **Select Models** (Sidebar)
   - Detection Model: Choose vehicle detection model (default: Exp23 Best)
   - Segmentation Model: Choose water segmentation model
   - Confidence Threshold: Adjust detection sensitivity (default: 0.10)

2. **Upload Images** (Tab 1: Upload & Process)
   - Click "Browse files" to upload flood images
   - Click "🚀 Process Images" to run detection and segmentation
   - Use radio buttons to switch between views:
     - Detection Only
     - Segmentation Only
     - Combined

3. **View Statistics** (Tab 2: Statistics)
   - See overall statistics across all processed images
   - View per-image breakdown

4. **Download Results** (Tab 3: Download Results)
   - Download all results as ZIP
   - Or download individual results per image

---

## 📁 Project Structure

```
Project3/
├── app.py                      # Main Streamlit web interface
├── detection.py                # Vehicle part detection (body/window/wheel)
├── segmentation.py             # Water segmentation
├── utils.py                    # Submersion calculation and visualization
├── config.py                   # Model paths and configuration
├── run.bat                     # Quick launch script
│
├── models/                     # Trained YOLO models
│   ├── vehical/                # Vehicle part detection models
│   │   ├── best_exp23.pt      # Best performing model (12.37% max conf)
│   │   └── last_exp23.pt      # Last epoch model
│   └── flood/                  # Water segmentation models
│       ├── best.pt            # Best performing model
│       └── last.pt            # Last epoch model
│
├── demo_images/                # Sample flood images for testing
├── test/                       # Test scripts and batch files
└── README/                     # Detailed documentation
    ├── README.md              # Complete system documentation
    ├── requirement.md         # Project requirements
    ├── project_details.md     # Implementation details
    ├── architecture.md        # System architecture
    ├── CUSTOM_MODELS.md       # Custom model information
    ├── CUSTOM_MODEL_CLASSES.md # Model class definitions
    └── SETUP_COMPLETE.md      # Setup verification guide
```

---

## 🎯 Key Features

### Custom Part-Based Detection
- Detects **body**, **window**, and **wheel** separately
- Analyzes submersion level per part:
  - Wheels underwater → Partially Submerged
  - Body/Window underwater → Fully Submerged

### Water Segmentation
- Segments water areas in flood images
- Calculates water coverage percentage
- Bright blue overlay (60% opacity) for visibility

### Submersion Analysis
- **Fully Submerged**: ≥70% of body/window underwater
- **Partially Submerged**: Wheels underwater or 20-70% body coverage
- **Not Submerged**: <20% water coverage

### Color-Coded Visualization
- 🔴 **Red**: Fully Submerged
- 🟠 **Orange**: Partially Submerged
- 🟢 **Green**: Not Submerged
- 🔵 **Blue**: Water Area

---

## ⚠️ Important Notes

### Model Performance
The vehicle detection model (exp23) was trained on **ground-level clear vehicle photos**, not aerial flood images. For best results:

1. ✅ **Recommended**: Use images from the training dataset
   - Location: `机器学习课程实践\project1\dataset2\images\test\`
   - These images will give accurate detections

2. ⚠️ **Limited**: Using new flood images (like cc*.jpg)
   - Detection may be poor due to different angles, lighting, and conditions
   - Consider retraining with flood-specific images

3. 🔄 **Alternative**: Use official YOLO models (YOLOv11n) for general vehicle detection

### Confidence Threshold
- Default: **0.10** (10%)
- Range: 0.01 to 1.0
- Lower values = more detections (but more false positives)
- Higher values = fewer detections (but higher accuracy)

---

## 📊 Sample Output

For each processed image, the system provides:

1. **Visual Results**
   - Detection boxes with color-coded submersion levels
   - Water segmentation overlay (bright blue)
   - Combined view showing both

2. **Statistics**
   - Total Vehicles/Parts Detected
   - Fully Submerged Count
   - Partially Submerged Count
   - Not Submerged Count

3. **Detailed Information**
   - Per-part submersion ratio
   - Bounding box coordinates
   - Detection confidence scores

---

## 🔧 Configuration

Edit `config.py` to:
- Add/remove model options
- Change default confidence thresholds
- Modify custom class definitions

---

## 📚 Detailed Documentation

For more information, see the **README/** folder:

- **README.md**: Complete system documentation
- **requirement.md**: Original project requirements
- **architecture.md**: Technical architecture details
- **CUSTOM_MODELS.md**: How custom models work
- **CUSTOM_MODEL_CLASSES.md**: Class definitions and training info

---

## 🐛 Troubleshooting

### No Detections?
- Lower confidence threshold (try 0.05)
- Use images from training dataset
- Check if models loaded correctly (console output)

### Water Not Visible?
- Water overlay is 60% opacity with bright blue color
- Check if segmentation model detected water (see console output)
- Verify water exists in the image

### Results Disappear?
- Make sure you're in the "Upload & Process" tab
- Results persist across view switches (Detection/Segmentation/Combined)

---

## 📝 Academic Context

**Project**: Task 4 - Vehicle Submersion Detection System
**Course**: Machine Learning Practice (机器学习课程实践)
**Due Date**: October 28, 2025
**Environment**: Conda (yolov11), Python 3.x, PyTorch with CUDA

---

## 🎓 Credits

Built with:
- **Ultralytics YOLO** (v8 & v11) - Object detection and segmentation
- **Streamlit** - Web interface
- **OpenCV** - Image processing
- **PyTorch** - Deep learning backend

---

**🚀 Ready to start? Run:** `streamlit run app.py`
