# System Architecture Documentation
# 水积识别和车辆淹没部位判别系统 - 架构文档

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Module Descriptions](#module-descriptions)
4. [Data Flow](#data-flow)
5. [Model Information](#model-information)
6. [Code Structure](#code-structure)
7. [Technical Implementation](#technical-implementation)

---

## 🏗️ System Overview

This is a **modular web-based application** built with Streamlit for detecting vehicles in flooded areas and analyzing their submersion levels. The system uses YOLO models for both object detection and semantic segmentation.

### Technology Stack
- **Frontend**: Streamlit (Web UI)
- **Backend**: Python 3.8+
- **Deep Learning**: Ultralytics YOLO (v8 & v11)
- **Computer Vision**: OpenCV
- **Framework**: PyTorch (with CUDA support)

### Key Characteristics
- **Modular Design**: Separated concerns (detection, segmentation, visualization, statistics)
- **Model Agnostic**: Support multiple YOLO versions
- **Real-time Processing**: Immediate feedback on image upload
- **Scalable**: Can process single or batch images

---

## 📊 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE (Streamlit)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Upload &   │  │  Statistics  │  │   Download   │     │
│  │   Process    │  │   Dashboard  │  │   Results    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│                     MAIN APPLICATION (app.py)                │
│  - Session Management                                        │
│  - Model Configuration                                       │
│  - Result Storage                                            │
└────────────┬────────────────────────────────────────────────┘
             │
       ┌─────┴─────┐
       ▼           ▼
┌──────────┐  ┌──────────────┐
│ Detection│  │ Segmentation │
│ Module   │  │   Module     │
│          │  │              │
│ YOLO     │  │  YOLO-Seg    │
│ Models   │  │   Models     │
└────┬─────┘  └─────┬────────┘
     │              │
     └──────┬───────┘
            ▼
    ┌───────────────┐
    │ Utils Module  │
    │               │
    │ - Statistics  │
    │ - Visualize   │
    │ - Analysis    │
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │  File System  │
    │               │
    │ - uploads/    │
    │ - results/    │
    │ - models/     │
    └───────────────┘
```

---

## 📦 Module Descriptions

### 1. `app.py` - Main Application
**Purpose**: Entry point and UI orchestration

**Key Functions**:
- `main()`: Initialize and run the Streamlit application
- `process_images()`: Orchestrate detection and segmentation pipeline
- `display_result()`: Render results with split-screen view
- `display_statistics()`: Show aggregate statistics
- `display_download_section()`: Handle file downloads
- `create_results_zip()`: Package results for bulk download

**Responsibilities**:
- UI layout and navigation (tabs, sidebar)
- Model selection interface
- Session state management
- File upload handling
- Result display and download

**Key Technologies**:
- Streamlit for web UI
- PIL for image handling
- Session state for persistence

---

### 2. `detection.py` - Vehicle Detection Module
**Purpose**: Detect vehicles using YOLO object detection models

**Class**: `VehicleDetector`

**Key Methods**:
```python
__init__(model_name, conf_threshold)     # Initialize detector
detect(image)                            # Run detection on image
visualize(image, detections)             # Draw bounding boxes
get_vehicle_count(detections)            # Count detected vehicles
get_vehicle_by_type(detections)          # Group by vehicle type
switch_model(new_model_name)             # Change detection model
```

**Vehicle Classes Detected** (COCO Dataset):
- Class 2: Car
- Class 3: Motorcycle
- Class 5: Bus
- Class 7: Truck

**Detection Output Format**:
```python
{
    'boxes': [[x1, y1, x2, y2], ...],      # Bounding boxes
    'scores': [0.95, 0.87, ...],           # Confidence scores
    'class_ids': [2, 2, 7, ...],           # COCO class IDs
    'class_names': ['car', 'car', 'truck'] # Human-readable names
}
```

**Processing Flow**:
```
Image → YOLO Model → Raw Results → Filter Vehicles → Parsed Detections
```

---

### 3. `segmentation.py` - Water Segmentation Module
**Purpose**: Identify water/flood regions using YOLO segmentation models

**Class**: `WaterSegmentation`

**Key Methods**:
```python
__init__(model_name, conf_threshold)     # Initialize segmenter
segment(image)                           # Run segmentation
visualize(image, segmentation)           # Draw masks and contours
get_water_mask(segmentation)             # Extract water regions
get_segmentation_stats(segmentation)     # Calculate coverage stats
switch_model(new_model_name)             # Change segmentation model
```

**Segmentation Output Format**:
```python
{
    'masks': [mask1, mask2, ...],         # Binary masks (H×W)
    'boxes': [[x1, y1, x2, y2], ...],     # Bounding boxes
    'scores': [0.92, 0.88, ...],          # Confidence scores
    'class_ids': [15, 15, ...],           # Class IDs
    'class_names': ['water', ...],        # Class names
    'combined_mask': combined_mask        # Union of all masks
}
```

**Processing Flow**:
```
Image → YOLO-Seg Model → Masks → Resize to Original → Combined Mask
```

---

### 4. `utils.py` - Utility Functions
**Purpose**: Core analysis, statistics, and visualization logic

**Key Functions**:

#### `calculate_submersion_stats(image, detection_results, segmentation_results)`
- Analyzes overlap between vehicle boxes and water masks
- Calculates submersion ratio for each vehicle
- Classifies vehicles into submersion levels
- Returns detailed statistics

**Submersion Level Thresholds**:
- **Fully Submerged**: ≥70% overlap with water
- **Partially Submerged**: 20-70% overlap with water
- **Not Submerged**: <20% overlap with water

**Algorithm**:
```
For each detected vehicle:
    1. Create vehicle mask from bounding box
    2. Calculate overlap with water mask
    3. Compute submersion_ratio = overlap_pixels / vehicle_pixels
    4. Classify based on thresholds
    5. Analyze which parts are submerged (top, middle, bottom)
```

#### `analyze_vehicle_parts(vehicle_mask, water_mask, vehicle_height)`
- Divides vehicle into thirds (top, middle, bottom)
- Checks which parts intersect with water
- Identifies if wheels/bottom are submerged

#### `visualize_results(image, detection_results, segmentation_results, ...)`
- Creates color-coded visualization
- Draws bounding boxes with submersion level colors
- Overlays water segmentation
- Adds labels with submersion percentages

**Color Coding**:
```python
Red (255, 0, 0)       → Fully Submerged
Orange (255, 165, 0)  → Partially Submerged
Green (0, 255, 0)     → Not Submerged
Blue (30, 144, 255)   → Water Area
```

#### `create_split_view(original_image, result_image, vertical=False)`
- Creates side-by-side comparison
- Supports horizontal or vertical split

#### `add_legend(image, position='bottom-right')`
- Adds color legend to visualization
- Explains submersion level colors

---

### 5. `config.py` - Configuration
**Purpose**: Centralized configuration management

**Configuration Categories**:
- Directory paths
- Model names and mappings
- Detection parameters (thresholds)
- Submersion level thresholds
- Color schemes
- UI settings
- File upload settings

**Why Separate Config?**
- Easy to modify parameters without touching code
- Single source of truth
- Maintainability

---

### 6. `test_system.py` - System Verification
**Purpose**: Validate system setup before running

**Test Categories**:
1. **Import Tests**: Verify all dependencies are installed
2. **Module Tests**: Check custom modules load correctly
3. **Directory Tests**: Ensure required folders exist
4. **Model Loading Tests**: Verify models can be downloaded and loaded

**Usage**:
```bash
python test_system.py
```

---

## 🔄 Data Flow

### Complete Processing Pipeline

```
1. USER UPLOADS IMAGE
   ↓
2. IMAGE SAVED TO uploads/
   ↓
3. PARALLEL PROCESSING:
   ├─→ VehicleDetector.detect(image)
   │   └─→ Returns: boxes, classes, scores
   │
   └─→ WaterSegmentation.segment(image)
       └─→ Returns: masks, combined_mask
   ↓
4. SUBMERSION ANALYSIS
   - calculate_submersion_stats()
   - For each vehicle:
     * Calculate overlap with water
     * Determine submersion level
     * Analyze submerged parts
   ↓
5. VISUALIZATION
   - visualize_results()
   - Create 3 versions:
     * Detection only
     * Segmentation only
     * Combined
   ↓
6. SAVE RESULTS
   - Detection image → results/
   - Segmentation image → results/
   - Combined image → results/
   - Statistics JSON → results/
   ↓
7. DISPLAY IN UI
   - Split-screen view
   - Statistics metrics
   - Download options
   ↓
8. USER DOWNLOADS
   - Individual images (PNG)
   - Statistics (JSON)
   - Bulk ZIP file
```

### Session State Management

```python
st.session_state.processed_results = [
    {
        'filename': 'image1.jpg',
        'timestamp': '20251027_135530',
        'original_image': np.array(...),
        'detection_image': np.array(...),
        'segmentation_image': np.array(...),
        'combined_image': np.array(...),
        'stats': {...},
        'original_path': 'uploads/...',
        'result_prefix': 'results/...'
    },
    ...
]
```

---

## 🤖 Model Information

### Do You Need Pre-trained .pt Files?

**Short Answer: NO! ❌**

The system uses Ultralytics YOLO, which **automatically downloads** models from the official repository on first use.

### How It Works

1. **First Run**:
   ```python
   detector = VehicleDetector('yolo11n.pt')
   # Ultralytics checks if yolo11n.pt exists locally
   # If not found, downloads from official repo
   # Saves to: ~/.ultralytics/weights/ or models/
   ```

2. **Subsequent Runs**:
   ```python
   # Model is cached locally, loads instantly
   ```

### Model Download Details

**Detection Models** (Automatically Downloaded):
| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| yolo11n.pt | ~3 MB | Fastest | Good | Testing, demos |
| yolo11s.pt | ~10 MB | Fast | Better | Balanced |
| yolo11m.pt | ~25 MB | Medium | High | Production |
| yolo11l.pt | ~50 MB | Slow | Highest | Maximum accuracy |
| yolov8n.pt | ~3 MB | Fastest | Good | Alternative |
| yolov8s.pt | ~10 MB | Fast | Better | Alternative |

**Segmentation Models** (Automatically Downloaded):
| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| yolo11n-seg.pt | ~3 MB | Fastest | Good | Testing, demos |
| yolo11s-seg.pt | ~10 MB | Fast | Better | Balanced |
| yolo11m-seg.pt | ~25 MB | Medium | High | Production |
| yolov8n-seg.pt | ~3 MB | Fastest | Good | Alternative |
| yolov8s-seg.pt | ~10 MB | Fast | Better | Alternative |

### Custom Trained Models (Optional)

**If you have custom-trained .pt files**:

1. **Place in models/ directory**:
   ```
   models/
   ├── my_custom_detector.pt
   └── my_custom_segmenter.pt
   ```

2. **Modify config.py**:
   ```python
   DETECTION_MODELS = {
       "YOLOv11n": "yolo11n.pt",
       "My Custom Model": "models/my_custom_detector.pt",  # Add this
   }
   ```

3. **The system will use your custom model**:
   - No code changes needed in detection.py or segmentation.py
   - YOLO automatically detects local .pt files
   - Can switch between official and custom models in UI

### Model Loading Priority

```
1. Check: models/model_name.pt (local directory)
   ↓
2. Check: ~/.ultralytics/weights/model_name.pt (cache)
   ↓
3. Download: From Ultralytics GitHub repository
   ↓
4. Cache: Save to ~/.ultralytics/weights/
```

---

## 💻 Code Structure

### File Organization

```
Project3/
│
├── Core Application Files
│   ├── app.py                 # Main Streamlit app (UI + orchestration)
│   ├── detection.py           # Vehicle detection logic
│   ├── segmentation.py        # Water segmentation logic
│   └── utils.py               # Analysis & visualization utilities
│
├── Configuration & Setup
│   ├── config.py              # System configuration
│   ├── requirements.txt       # Python dependencies
│   └── .gitignore            # Git ignore rules
│
├── Documentation
│   ├── README.md              # User guide
│   ├── architecture.md        # This file
│   ├── project_details.md     # Requirements breakdown
│   └── requirement.md         # Original requirements (Chinese)
│
├── Scripts
│   ├── run.bat               # Windows launcher
│   └── test_system.py        # System verification
│
└── Data Directories
    ├── uploads/              # User-uploaded images
    ├── results/              # Processed results
    ├── models/               # Downloaded YOLO models (auto-created)
    └── temp/                 # Temporary files
```

### Module Dependencies

```
app.py
├── imports: detection.VehicleDetector
├── imports: segmentation.WaterSegmentation
├── imports: utils.calculate_submersion_stats
├── imports: utils.visualize_results
└── imports: utils.create_split_view

detection.py
├── imports: ultralytics.YOLO
└── imports: cv2, numpy

segmentation.py
├── imports: ultralytics.YOLO
└── imports: cv2, numpy

utils.py
└── imports: cv2, numpy
```

### Execution Flow

```
1. User runs: streamlit run app.py
   ↓
2. app.py calls main()
   ↓
3. Streamlit renders UI
   ↓
4. User selects models in sidebar
   ↓
5. User uploads images
   ↓
6. User clicks "Process Images"
   ↓
7. process_images() called:
   ├─→ Initialize VehicleDetector
   ├─→ Initialize WaterSegmentation
   ├─→ For each image:
   │   ├─→ detector.detect(image)
   │   ├─→ segmenter.segment(image)
   │   ├─→ calculate_submersion_stats()
   │   ├─→ visualize_results() × 3
   │   └─→ Save results
   └─→ Store in session_state
   ↓
8. display_result() renders UI
   ↓
9. User views statistics, downloads results
```

---

## 🔧 Technical Implementation

### Key Design Patterns

#### 1. **Separation of Concerns**
- **Detection logic** isolated in detection.py
- **Segmentation logic** isolated in segmentation.py
- **Analysis logic** isolated in utils.py
- **UI logic** isolated in app.py

**Benefit**: Easy to modify or replace individual components

#### 2. **Model Abstraction**
```python
class VehicleDetector:
    def __init__(self, model_name, conf_threshold):
        self.model = YOLO(model_name)  # Abstraction layer

    def detect(self, image):
        # Standard interface regardless of model
        return results
```

**Benefit**: Switch models without changing calling code

#### 3. **Session State Pattern**
```python
# Persist data across Streamlit reruns
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []

# Access anywhere in app
results = st.session_state.processed_results
```

**Benefit**: Maintain state in stateless Streamlit environment

#### 4. **Factory Pattern** (Implicit)
```python
# app.py
detector = VehicleDetector(model_name, conf_threshold)
segmenter = WaterSegmentation(model_name, conf_threshold)
```

**Benefit**: Flexible instantiation with different configurations

### Algorithms

#### Submersion Calculation Algorithm

```python
def calculate_submersion_ratio(vehicle_box, water_mask):
    """
    Calculate what percentage of vehicle is underwater

    Algorithm:
    1. Extract vehicle bounding box coordinates
    2. Create binary mask for vehicle region
    3. Apply logical AND with water mask
    4. Count overlapping pixels
    5. Divide by total vehicle pixels
    """
    x1, y1, x2, y2 = vehicle_box

    # Step 1: Create vehicle mask
    vehicle_mask = np.zeros_like(water_mask)
    vehicle_mask[y1:y2, x1:x2] = 1

    # Step 2: Find overlap
    overlap = np.logical_and(vehicle_mask, water_mask)

    # Step 3: Calculate ratio
    overlap_pixels = np.sum(overlap)
    vehicle_pixels = np.sum(vehicle_mask)

    ratio = overlap_pixels / vehicle_pixels if vehicle_pixels > 0 else 0

    return ratio
```

**Time Complexity**: O(H × W) where H, W are image dimensions
**Space Complexity**: O(H × W) for temporary masks

#### Vehicle Part Analysis Algorithm

```python
def analyze_vehicle_parts(vehicle_mask, water_mask, vehicle_height):
    """
    Determine which parts of vehicle are submerged

    Algorithm:
    1. Divide vehicle height into 3 equal parts
    2. For each part (top, middle, bottom):
       - Calculate water overlap in that region
       - If >30% submerged, mark as True
    3. Bottom region also indicates wheel submersion
    """
    third_height = vehicle_height // 3

    parts = {'top': False, 'middle': False, 'bottom': False, 'wheels': False}

    # Check each third
    for part_name, start, end in [
        ('top', 0, third_height),
        ('middle', third_height, 2*third_height),
        ('bottom', 2*third_height, vehicle_height)
    ]:
        region_overlap = calculate_overlap(
            vehicle_mask[start:end, :],
            water_mask[start:end, :]
        )

        if region_overlap > 0.3:
            parts[part_name] = True
            if part_name == 'bottom':
                parts['wheels'] = True

    return parts
```

### Performance Considerations

#### 1. **Model Loading**
- Models loaded once per session
- Cached in memory
- Switching models reloads (< 1 second for small models)

#### 2. **Image Processing**
- OpenCV for fast numpy operations
- GPU acceleration via PyTorch (if CUDA available)
- Batch processing not implemented (sequential for clarity)

#### 3. **Memory Management**
- Images stored in session state (RAM)
- Consider clearing old results for long sessions
- Typical usage: 10-50 images = 100-500 MB RAM

#### 4. **Optimization Opportunities**
```python
# Current: Sequential processing
for image in images:
    detect(image)
    segment(image)

# Future: Batch processing (faster)
results = model.predict(images, batch=True)
```

### Error Handling

#### Model Loading Errors
```python
try:
    self.model = YOLO(model_name)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # Fall back to default model or show error
```

#### Image Processing Errors
```python
try:
    results = detector.detect(image)
except Exception as e:
    st.error(f"Detection failed: {e}")
    # Continue with next image
```

#### File I/O Errors
```python
try:
    image.save(path)
except IOError as e:
    st.warning(f"Could not save file: {e}")
    # Continue processing, just don't save
```

---

## 🚀 Extending the System

### Adding New Models

**1. Official YOLO Models**:
```python
# config.py
DETECTION_MODELS = {
    "YOLOv11x": "yolo11x.pt",  # Add new model
}
```

**2. Custom Trained Models**:
```python
# Train your model
yolo train data=flood_data.yaml model=yolo11n.pt epochs=100

# Use in system
DETECTION_MODELS = {
    "Flood Custom": "models/flood_custom.pt",
}
```

### Adding New Features

**Example: Export to PDF**
```python
# In app.py
def export_to_pdf(results):
    from fpdf import FPDF
    pdf = FPDF()
    # Add images and statistics
    return pdf

# Add download button
pdf = export_to_pdf(st.session_state.processed_results)
st.download_button("Download PDF", pdf, "results.pdf")
```

### Modifying Submersion Thresholds

```python
# config.py
FULLY_SUBMERGED_THRESHOLD = 0.8  # Change from 0.7 to 0.8
PARTIALLY_SUBMERGED_THRESHOLD = 0.3  # Change from 0.2 to 0.3
```

---

## 📚 Summary

### System Characteristics
- ✅ **Modular**: Each component has single responsibility
- ✅ **Extensible**: Easy to add models, features, or export formats
- ✅ **User-friendly**: Web interface with clear visual feedback
- ✅ **Automated**: Models download automatically
- ✅ **Flexible**: Support multiple model versions
- ✅ **Production-ready**: Error handling, documentation, testing

### No Pre-trained Files Needed!
The system uses **official Ultralytics YOLO models** that download automatically. You only need custom .pt files if you've trained specialized models for your specific use case (e.g., trained specifically on flood/water imagery).

### Recommended Workflow
1. Start with lightweight models (yolo11n, yolo11n-seg)
2. Test with sample images
3. If accuracy insufficient, upgrade to larger models (yolo11m, yolo11l)
4. Optionally: Train custom models on flood-specific dataset
5. Deploy for demonstration on October 28th

---

**Questions?** Check README.md for usage guide or run `python test_system.py` to verify setup!
