"""
Test all images in images_test folder
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
print("TESTING ALL IMAGES IN images_test/")
print("="*60)

# Find all test images
test_images = list(Path("images_test").rglob("*.jpg")) + list(Path("images_test").rglob("*.png"))

if not test_images:
    print("\n❌ No test images found")
    sys.exit(1)

print(f"\nFound {len(test_images)} test images\n")

# Load models
detector = VehicleDetector('models/vehical/best.pt', conf_threshold=0.1)
segmenter = WaterSegmentation('models/flood/best.pt', conf_threshold=0.1)

# Test each image
results_summary = []

for img_path in test_images[:10]:  # Test first 10 images
    print(f"📸 {img_path.name}")

    image = cv2.imread(str(img_path))
    if image is None:
        print("   ❌ Could not read\n")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    # Detect
    det_results = detector.detect(image_rgb)
    seg_results = segmenter.segment(image_rgb)

    n_vehicles = len(det_results['boxes'])
    n_water = len(seg_results['masks'])

    if n_vehicles > 0:
        vehicle_types = set(det_results['class_names'])
        print(f"   ✅ VEHICLES: {n_vehicles} ({', '.join(vehicle_types)})")
    else:
        print(f"   ⚪ No vehicles detected")

    if n_water > 0:
        water_pct = np.sum(seg_results['combined_mask'] > 0) / seg_results['combined_mask'].size * 100
        print(f"   ✅ WATER: {n_water} segments ({water_pct:.1f}% coverage)")
    else:
        print(f"   ⚪ No water detected")

    print(f"   Image size: {w}x{h}\n")

    results_summary.append({
        'name': img_path.name,
        'vehicles': n_vehicles,
        'water': n_water,
        'size': f"{w}x{h}"
    })

print("="*60)
print("SUMMARY")
print("="*60)

for r in results_summary:
    status = "✅" if r['vehicles'] > 0 or r['water'] > 0 else "❌"
    print(f"{status} {r['name']:30s} - Vehicles: {r['vehicles']}, Water: {r['water']}")

print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)

has_vehicles = any(r['vehicles'] > 0 for r in results_summary)
has_water = any(r['water'] > 0 for r in results_summary)

if has_vehicles and has_water:
    print("\n✅ Found images with both vehicles and water!")
    print("Use these images in the web app for best demo results.")
elif has_water and not has_vehicles:
    print("\n⚠️  Water detection works, but no vehicles detected in any image.")
    print("\nPossible reasons:")
    print("1. These images don't have vehicles")
    print("2. Vehicles are too different from training data")
    print("3. Images need to be from your training dataset")
    print("\n💡 Solution:")
    print("- Use images from your TRAINING dataset (the ones you trained on)")
    print("- Make sure images have visible vehicle parts (body, window, wheel)")
else:
    print("\n⚠️  Models not detecting on these test images.")
    print("\n💡 Use images from your training dataset!")
