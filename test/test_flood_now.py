"""
Test the current flood segmentation models
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import cv2
import numpy as np
from segmentation import WaterSegmentation

# Test with a flood image
test_img = "demo_images/cc172.jpg"

print("="*60)
print("TESTING CURRENT FLOOD MODELS")
print("="*60)

image = cv2.imread(test_img)
if image is None:
    test_img = r"C:\Users\xwj\Desktop\study\YOLO\机器学习课程实践\project1\dataset2\images\test\cc172.jpg"
    image = cv2.imread(test_img)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(f"\n📸 Test image: {test_img}")
print(f"   Shape: {image_rgb.shape}")

# Test both best.pt and last.pt
models_to_test = [
    ("best.pt", "models/flood/best.pt"),
    ("last.pt", "models/flood/last.pt"),
]

for name, path in models_to_test:
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    segmenter = WaterSegmentation(path, conf_threshold=0.25)

    results = segmenter.segment(image_rgb)

    n_segments = len(results['masks'])
    print(f"\n   Segments found: {n_segments}")

    if n_segments > 0:
        print(f"   Classes: {results['class_names']}")

        water_pixels = np.sum(results['combined_mask'] > 0)
        total_pixels = results['combined_mask'].size
        coverage = (water_pixels / total_pixels) * 100

        print(f"   Water coverage: {coverage:.1f}%")

        # Check scores
        scores = results['scores']
        print(f"   Confidence: min={min(scores):.2%}, max={max(scores):.2%}, avg={sum(scores)/len(scores):.2%}")

        print(f"\n   ✅ This model WORKS!")
    else:
        print(f"\n   ❌ NO water segments detected")

        # Try with lower confidence
        print(f"\n   Trying conf=0.1...")
        segmenter.conf_threshold = 0.1
        results_low = segmenter.segment(image_rgb)

        if len(results_low['masks']) > 0:
            print(f"   Found {len(results_low['masks'])} segments at conf=0.1")
        else:
            print(f"   Still nothing at conf=0.1")

print(f"\n{'='*60}")
print("CONCLUSION")
print(f"{'='*60}")
print("\nIf both models show water segments, they're working!")
print("If not, you need your partner's actual trained model files.")
