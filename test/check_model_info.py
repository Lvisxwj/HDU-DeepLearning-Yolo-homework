"""
Check detailed model information
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from ultralytics import YOLO
import torch

print("="*60)
print("MODEL INFORMATION")
print("="*60)

# Load your vehicle model
model = YOLO('models/vehical/best.pt')

print(f"\nModel path: models/vehical/best.pt")
print(f"Task: {model.task}")
print(f"Model type: {type(model.model)}")

# Check training info
if hasattr(model.model, 'args'):
    print(f"\nTraining args:")
    print(f"  Image size: {model.model.args.get('imgsz', 'Unknown')}")
    print(f"  Batch: {model.model.args.get('batch', 'Unknown')}")
    print(f"  Epochs: {model.model.args.get('epochs', 'Unknown')}")

# Class names
print(f"\nClasses ({len(model.names)}):")
for idx, name in model.names.items():
    print(f"  {idx}: {name}")

# Check if model is in eval mode
print(f"\nModel mode: {'eval' if not model.model.training else 'training'}")

# Test prediction with different parameters
print("\n" + "="*60)
print("TESTING PREDICTION PARAMETERS")
print("="*60)

import cv2
import numpy as np

# Read test image
image = cv2.imread('images_test/images/cc1.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(f"\nTest image shape: {image_rgb.shape}")

# Try different image sizes
for imgsz in [640, 320, 1280]:
    print(f"\n--- Image size: {imgsz} ---")
    for conf in [0.01, 0.1, 0.25]:
        results = model(image_rgb, imgsz=imgsz, conf=conf, verbose=False)
        n_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        print(f"  conf={conf}: {n_detections} detections")

# Try with augmentation
print(f"\n--- With augmentation ---")
results = model(image_rgb, augment=True, conf=0.1, verbose=False)
n_detections = len(results[0].boxes) if results[0].boxes is not None else 0
print(f"  Augmented: {n_detections} detections")

# Check model confidence distribution
print("\n" + "="*60)
print("CHECKING MODEL OUTPUT CONFIDENCE")
print("="*60)

# Run with conf=0 to see ALL predictions
results = model(image_rgb, conf=0.001, verbose=False)[0]

if results.boxes is not None and len(results.boxes) > 0:
    scores = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)

    print(f"\nTotal predictions (conf>0.001): {len(scores)}")
    print(f"Max confidence: {scores.max():.4f}")
    print(f"Min confidence: {scores.min():.4f}")
    print(f"Mean confidence: {scores.mean():.4f}")

    # Group by class
    for cls_id in [0, 1, 2]:
        cls_scores = scores[class_ids == cls_id]
        if len(cls_scores) > 0:
            print(f"\nClass {cls_id} ({model.names[cls_id]}):")
            print(f"  Count: {len(cls_scores)}")
            print(f"  Max conf: {cls_scores.max():.4f}")
            print(f"  Min conf: {cls_scores.min():.4f}")
else:
    print("\n❌ NO PREDICTIONS AT ALL (even with conf=0.001)")
    print("\nThis means the model is not detecting ANY vehicle parts.")
    print("\nPossible reasons:")
    print("1. Model was trained on very different images (different angle/zoom/quality)")
    print("2. Model is overfitted to specific training conditions")
    print("3. Image preprocessing mismatch")
    print("4. Model file might be corrupted")

    print("\n💡 Try:")
    print("1. Use images from your actual training set")
    print("2. Check if training images were aerial/top-down views like this")
    print("3. Retrain with more varied data if needed")
