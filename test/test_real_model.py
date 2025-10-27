"""
Test the ACTUAL trained model from runs_improved
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import cv2
from ultralytics import YOLO

# Test image from training dataset
test_img = r"C:\Users\xwj\Desktop\study\YOLO\机器学习课程实践\project1\dataset2\images\test\cc172.jpg"

print("="*60)
print("TESTING REAL TRAINED MODEL")
print("="*60)

# Test BOTH models
models_to_test = [
    ("Current (models/vehical/best.pt)", "models/vehical/best.pt"),
    ("Improved Run", r"C:\Users\xwj\Desktop\study\YOLO\机器学习课程实践\project1\scripts\runs_improved\exp_improved\weights\best.pt"),
    ("Exp23 Run", r"C:\Users\xwj\Desktop\study\YOLO\机器学习课程实践\project1\scripts\runs2\exp23\weights\best.pt"),
]

# Read image
image = cv2.imread(test_img)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(f"\n📸 Test image: cc172.jpg ({image_rgb.shape})")

for model_name, model_path in models_to_test:
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded")
        print(f"   Task: {model.task}")
        print(f"   Classes: {list(model.names.values())}")

        # Run detection
        results = model(image_rgb, conf=0.25, verbose=False)[0]

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            print(f"\n✅ DETECTIONS FOUND: {len(boxes)}")

            # Count by class
            for cls_id in set(class_ids):
                cls_name = model.names[cls_id]
                count = sum(1 for c in class_ids if c == cls_id)
                cls_scores = scores[class_ids == cls_id]
                print(f"   - {cls_name}: {count} (conf: {cls_scores.min():.2%} - {cls_scores.max():.2%})")

        else:
            print(f"\n❌ NO DETECTIONS")

            # Check with very low confidence
            results_low = model(image_rgb, conf=0.001, verbose=False)[0]
            if results_low.boxes is not None and len(results_low.boxes) > 0:
                max_conf = results_low.boxes.conf.max().item()
                print(f"   Max confidence found: {max_conf:.4%}")
            else:
                print(f"   No detections even at conf=0.001")

    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("\nIf all models show NO DETECTIONS or very low confidence,")
print("the training might not have converged properly.")
print("\nCheck your training logs/plots to verify training success.")
