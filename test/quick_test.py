"""
Quick test with a real image from images_test folder
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

print("="*60)
print("QUICK TEST WITH REAL IMAGE")
print("="*60)

# Find a test image
test_images = list(Path("images_test").rglob("*.jpg")) + list(Path("images_test").rglob("*.png"))

if not test_images:
    print("\n❌ No test images found in images_test/")
    print("Please put some test images in the images_test/ folder")
    sys.exit(1)

test_image_path = test_images[0]
print(f"\n📸 Using test image: {test_image_path}")

# Read image
image = cv2.imread(str(test_image_path))
if image is None:
    print(f"❌ Could not read image: {test_image_path}")
    sys.exit(1)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(f"✅ Image loaded: {image_rgb.shape}")

# Test detection with different confidence levels
print("\n" + "="*60)
print("TESTING VEHICLE DETECTION")
print("="*60)

detector = VehicleDetector('models/vehical/best.pt', conf_threshold=0.25)

for conf in [0.1, 0.25, 0.5]:
    detector.conf_threshold = conf
    results = detector.detect(image_rgb)
    print(f"\nConfidence {conf}: Found {len(results['boxes'])} objects")
    if results['class_names']:
        print(f"  Classes: {results['class_names']}")
        for i, (box, cls, score) in enumerate(zip(results['boxes'], results['class_names'], results['scores'])):
            print(f"  [{i}] {cls}: {score:.3f} at {box}")

# Test segmentation
print("\n" + "="*60)
print("TESTING WATER SEGMENTATION")
print("="*60)

segmenter = WaterSegmentation('models/flood/best.pt', conf_threshold=0.25)

for conf in [0.1, 0.25, 0.5]:
    segmenter.conf_threshold = conf
    results = segmenter.segment(image_rgb)
    print(f"\nConfidence {conf}: Found {len(results['masks'])} segments")
    if results['class_names']:
        print(f"  Classes: {results['class_names']}")
        water_pixels = np.sum(results['combined_mask'] > 0)
        total_pixels = results['combined_mask'].size
        print(f"  Water coverage: {water_pixels/total_pixels*100:.1f}%")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
print("\nIf you see 0 detections at all confidence levels,")
print("your test image might not contain vehicles or water.")
print("Try with a different image from your training dataset!")
