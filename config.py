"""
Configuration file for the Vehicle Submersion Detection System
"""

# Directory paths
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
MODELS_DIR = "models"
TEMP_DIR = "temp"

# Model configurations
DETECTION_MODELS = {
    # Custom Trained Models (Your Models) - exp23 performs better
    "🎯 Vehicle Custom (Exp23 Best)": "models/vehical/best_exp23.pt",
    "🎯 Vehicle Custom (Exp23 Last)": "models/vehical/last_exp23.pt",
    "Vehicle Custom (Improved - Low Conf)": "models/vehical/best.pt",
    # Pre-trained YOLO Models
    "YOLOv11n": "yolo11n.pt",
    "YOLOv11s": "yolo11s.pt",
    "YOLOv11m": "yolo11m.pt",
    "YOLOv11l": "yolo11l.pt",
    "YOLOv8n": "yolov8n.pt",
    "YOLOv8s": "yolov8s.pt",
}

SEGMENTATION_MODELS = {
    # Custom Trained Models (Your Models)
    "🌊 Flood Custom (Best)": "models/flood/best.pt",
    "🌊 Flood Custom (Last)": "models/flood/last.pt",
    # Pre-trained YOLO Segmentation Models
    "YOLOv11n-seg": "yolo11n-seg.pt",
    "YOLOv11s-seg": "yolo11s-seg.pt",
    "YOLOv11m-seg": "yolo11m-seg.pt",
    "YOLOv8n-seg": "yolov8n-seg.pt",
    "YOLOv8s-seg": "yolov8s-seg.pt",
}

# Detection parameters
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
MIN_CONFIDENCE_THRESHOLD = 0.1
MAX_CONFIDENCE_THRESHOLD = 1.0

# Submersion level thresholds
FULLY_SUBMERGED_THRESHOLD = 0.7  # 70% or more
PARTIALLY_SUBMERGED_THRESHOLD = 0.2  # 20% or more

# Vehicle classes (COCO dataset - for official models)
COCO_VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

# Custom trained model classes (your models)
CUSTOM_VEHICLE_PARTS = {
    0: 'body',
    1: 'window',
    2: 'wheel'
}

CUSTOM_WATER_CLASSES = {
    0: 'water'
}

# Visualization colors (RGB)
COLORS = {
    'fully_submerged': (255, 0, 0),      # Red
    'partially_submerged': (255, 165, 0), # Orange
    'not_submerged': (0, 255, 0),        # Green
    'water': (30, 144, 255),             # Dodger Blue
}

# UI settings
PAGE_TITLE = "Vehicle Submersion Detection System"
PAGE_ICON = "🚗"
LAYOUT = "wide"

# File upload settings
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "bmp"]
MAX_FILE_SIZE_MB = 10

# Image processing
DEFAULT_IMAGE_SIZE = (640, 640)
VISUALIZATION_ALPHA = 0.5  # Transparency for overlays
