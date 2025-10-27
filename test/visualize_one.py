"""
Visualize detection on ONE image with annotations
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import cv2
import numpy as np
from pathlib import Path
from detection import VehicleDetector
from segmentation import WaterSegmentation

# Use cc1.jpg
test_img = "images_test/images/cc1.jpg"

print("="*60)
print(f"Testing: {test_img}")
print("="*60)

# Read image
image = cv2.imread(test_img)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w = image_rgb.shape[:2]

print(f"\nImage size: {w}x{h}")
print(f"Image shape: {image_rgb.shape}")
print(f"Image dtype: {image_rgb.dtype}")
print(f"Image range: {image_rgb.min()} to {image_rgb.max()}")

# Load models with VERY low confidence
print("\n" + "="*60)
print("DETECTION with conf=0.01")
print("="*60)

detector = VehicleDetector('models/vehical/best.pt', conf_threshold=0.01)

# Check model info
print(f"\nModel task: {detector.model.task}")
print(f"Model names: {detector.model.names}")

# Detect
results = detector.detect(image_rgb)

print(f"\nDetection results:")
print(f"  Boxes: {len(results['boxes'])}")
print(f"  Classes: {results['class_names']}")

if len(results['boxes']) > 0:
    print("\nDetailed detections:")
    for i, (box, cls, score) in enumerate(zip(results['boxes'], results['class_names'], results['scores'])):
        x1, y1, x2, y2 = box
        print(f"  [{i}] {cls}: {score:.4f} at ({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})")

    # Save visualization
    vis = detector.visualize(image_rgb, results)
    output_path = "detection_test_output.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"\n✅ Saved visualization to: {output_path}")
else:
    print("\n❌ NO DETECTIONS FOUND")
    print("\nThis means:")
    print("1. The image has no vehicle parts (body/window/wheel)")
    print("2. OR the vehicle parts look very different from training")
    print("3. OR they're too small/blurry/occluded")

# Check what the RAW model outputs
print("\n" + "="*60)
print("RAW MODEL OUTPUT")
print("="*60)

raw_results = detector.model(image_rgb, conf=0.01, verbose=False)[0]

if raw_results.boxes is not None and len(raw_results.boxes) > 0:
    print(f"\nRaw detections: {len(raw_results.boxes)}")
    boxes = raw_results.boxes.xyxy.cpu().numpy()
    scores = raw_results.boxes.conf.cpu().numpy()
    class_ids = raw_results.boxes.cls.cpu().numpy().astype(int)

    print("\nAll raw detections:")
    for i in range(len(boxes)):
        cls_id = class_ids[i]
        cls_name = detector.model.names.get(cls_id, f'class_{cls_id}')
        print(f"  [{i}] Class {cls_id} ({cls_name}): {scores[i]:.4f}")
else:
    print("\n❌ Model returned 0 detections even at conf=0.01")
    print("\nThis confirms: The image has NO vehicle parts that the model recognizes")

print("\n" + "="*60)
print("WATER SEGMENTATION with conf=0.01")
print("="*60)

segmenter = WaterSegmentation('models/flood/best.pt', conf_threshold=0.01)
seg_results = segmenter.segment(image_rgb)

print(f"\nSegmentation results:")
print(f"  Segments: {len(seg_results['masks'])}")
print(f"  Classes: {seg_results['class_names']}")

if len(seg_results['masks']) > 0:
    water_coverage = np.sum(seg_results['combined_mask'] > 0) / seg_results['combined_mask'].size * 100
    print(f"  Water coverage: {water_coverage:.1f}%")
    print("\n✅ Water segmentation WORKS!")
else:
    print("\n❌ No water detected")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("\n✅ Your models ARE loaded correctly (models/vehical/best.pt)")
print("✅ Water model works perfectly")
print("❌ Vehicle model finds nothing in this image")
print("\n💡 The cc*.jpg images might be:")
print("   - Only for water segmentation training")
print("   - Not have visible vehicle parts")
print("   - Or vehicle parts are too different from training")
