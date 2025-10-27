"""
Inspect custom model classes
This script will tell you what classes your models detect
"""

import sys
import io

# Set console encoding to UTF-8 for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from ultralytics import YOLO

print("="*60)
print("CUSTOM MODEL CLASS INSPECTION")
print("="*60)

print("\n" + "="*60)
print("Vehicle Detection Model - best.pt")
print("="*60)

try:
    model_vehicle = YOLO('models/vehical/best.pt')
    print(f"\nModel Type: {model_vehicle.task}")
    print(f"\nClass Names:")
    if hasattr(model_vehicle, 'names'):
        for idx, name in model_vehicle.names.items():
            print(f"  Class {idx}: {name}")
    print(f"\nTotal Classes: {len(model_vehicle.names)}")
except Exception as e:
    print(f"Error loading vehicle model: {e}")

print("\n" + "="*60)
print("Flood Segmentation Model - best.pt")
print("="*60)

try:
    model_flood = YOLO('models/flood/best.pt')
    print(f"\nModel Type: {model_flood.task}")
    print(f"\nClass Names:")
    if hasattr(model_flood, 'names'):
        for idx, name in model_flood.names.items():
            print(f"  Class {idx}: {name}")
    print(f"\nTotal Classes: {len(model_flood.names)}")
except Exception as e:
    print(f"Error loading flood model: {e}")

print("\n" + "="*60)
print("CONFIGURATION NEEDED")
print("="*60)
print("\nBased on the classes above, you need to:")
print("1. Update config.py with your custom class names")
print("2. Update detection.py if needed")
print("3. Update segmentation.py if needed")
print("\nRun: python create_custom_config.py")
