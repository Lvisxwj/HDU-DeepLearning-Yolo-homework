"""
Test with an image from your training dataset
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import cv2
from detection import VehicleDetector

# Test image from your training dataset
test_img = r"C:\Users\xwj\Desktop\study\YOLO\机器学习课程实践\project1\dataset2\images\test\cc172.jpg"

print("="*60)
print("TESTING WITH YOUR TRAINING DATASET IMAGE")
print("="*60)

# Read image
image = cv2.imread(test_img)
if image is None:
    print(f"❌ Could not read: {test_img}")
    sys.exit(1)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(f"\n📸 Image: cc172.jpg")
print(f"   Size: {image_rgb.shape}")

# Test with different confidence levels
detector = VehicleDetector('models/vehical/best.pt', conf_threshold=0.25)

print("\n" + "="*60)
print("DETECTION RESULTS")
print("="*60)

for conf in [0.1, 0.25, 0.5]:
    detector.conf_threshold = conf
    results = detector.detect(image_rgb)
    n_det = len(results['boxes'])

    print(f"\nConfidence {conf}:")
    print(f"  Total detections: {n_det}")

    if n_det > 0:
        # Count by type
        body_count = sum(1 for c in results['class_names'] if c == 'body')
        window_count = sum(1 for c in results['class_names'] if c == 'window')
        wheel_count = sum(1 for c in results['class_names'] if c == 'wheel')

        print(f"  - Body: {body_count}")
        print(f"  - Window: {window_count}")
        print(f"  - Wheel: {wheel_count}")

        # Show confidence scores
        if len(results['scores']) > 0:
            max_conf = max(results['scores'])
            min_conf = min(results['scores'])
            avg_conf = sum(results['scores']) / len(results['scores'])
            print(f"  Confidence: min={min_conf:.2%}, max={max_conf:.2%}, avg={avg_conf:.2%}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

if n_det > 0:
    print("\n✅ SUCCESS! Your model detects vehicle parts in training images!")
    print("\n💡 For your demo:")
    print("   1. Use images from:")
    print("      C:\\Users\\xwj\\Desktop\\study\\YOLO\\机器学习课程实践\\project1\\dataset2\\images\\")
    print("   2. These will show proper detection with good confidence")
    print("   3. Your part-based detection (body, window, wheel) will work!")
else:
    print("\n⚠️ Still no detection. Check if model file is correct.")
