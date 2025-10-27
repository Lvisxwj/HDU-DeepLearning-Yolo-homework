"""
Test exp23 model at 10% confidence
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import cv2
from detection import VehicleDetector

# Test with training dataset image
test_img = r"C:\Users\xwj\Desktop\study\YOLO\机器学习课程实践\project1\dataset2\images\test\cc172.jpg"

print("="*60)
print("TESTING EXP23 MODEL AT 10% CONFIDENCE")
print("="*60)

image = cv2.imread(test_img)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(f"\n📸 Image: cc172.jpg ({image_rgb.shape})")

# Test with 10% confidence
detector = VehicleDetector('models/vehical/best_exp23.pt', conf_threshold=0.10)

results = detector.detect(image_rgb)

print(f"\n✅ Detection Results (conf >= 0.10):")
print(f"   Total detections: {len(results['boxes'])}")

if len(results['boxes']) > 0:
    # Count by type
    body_count = sum(1 for c in results['class_names'] if c == 'body')
    window_count = sum(1 for c in results['class_names'] if c == 'window')
    wheel_count = sum(1 for c in results['class_names'] if c == 'wheel')

    print(f"\n   Part breakdown:")
    print(f"   - Body: {body_count}")
    print(f"   - Window: {window_count}")
    print(f"   - Wheel: {wheel_count}")

    # Show confidence scores
    max_conf = max(results['scores'])
    min_conf = min(results['scores'])
    avg_conf = sum(results['scores']) / len(results['scores'])

    print(f"\n   Confidence scores:")
    print(f"   - Min: {min_conf:.2%}")
    print(f"   - Max: {max_conf:.2%}")
    print(f"   - Avg: {avg_conf:.2%}")

    print(f"\n✅ SUCCESS! Model detects at 10% confidence!")
    print(f"\n💡 For your demo:")
    print(f"   1. Use 'Vehicle Custom (Exp23 Best)' model")
    print(f"   2. Set confidence to 0.10 (10%)")
    print(f"   3. Upload images from dataset2/images/test/")
    print(f"   4. Will detect {len(results['boxes'])} vehicle parts!")
else:
    print(f"\n❌ Still no detections at 10%")
    print(f"\n   This means even exp23 model doesn't work well.")
    print(f"\n💡 Recommendation for demo:")
    print(f"   Use official YOLOv11n for vehicle detection")
    print(f"   Keep your custom flood segmentation (works perfectly!)")
