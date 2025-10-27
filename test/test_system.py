"""
Test script to verify system setup
Run this to check if all modules are working correctly
"""

import sys
import numpy as np

def test_imports():
    """Test if all required packages are importable"""
    print("Testing imports...")

    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False

    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False

    try:
        from ultralytics import YOLO
        print("✅ Ultralytics imported successfully")
    except ImportError as e:
        print(f"❌ Ultralytics import failed: {e}")
        return False

    try:
        import torch
        print(f"✅ PyTorch imported successfully (version: {torch.__version__})")
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available, will use CPU")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False

    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False

    try:
        from PIL import Image
        print("✅ PIL imported successfully")
    except ImportError as e:
        print(f"❌ PIL import failed: {e}")
        return False

    return True

def test_modules():
    """Test if custom modules are importable"""
    print("\nTesting custom modules...")

    try:
        from detection import VehicleDetector
        print("✅ detection.py module loaded")
    except ImportError as e:
        print(f"❌ detection.py import failed: {e}")
        return False

    try:
        from segmentation import WaterSegmentation
        print("✅ segmentation.py module loaded")
    except ImportError as e:
        print(f"❌ segmentation.py import failed: {e}")
        return False

    try:
        from utils import calculate_submersion_stats, visualize_results
        print("✅ utils.py module loaded")
    except ImportError as e:
        print(f"❌ utils.py import failed: {e}")
        return False

    return True

def test_directories():
    """Test if required directories exist"""
    print("\nTesting directories...")

    from pathlib import Path

    dirs = ['uploads', 'results', 'models', 'temp']
    all_exist = True

    for dir_name in dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory missing")
            all_exist = False

    return all_exist

def test_model_loading():
    """Test if models can be loaded"""
    print("\nTesting model loading (this may take a while on first run)...")

    try:
        from detection import VehicleDetector
        print("Loading detection model...")
        detector = VehicleDetector('yolo11n.pt', conf_threshold=0.25)
        print("✅ Detection model loaded successfully")

        # Test with dummy image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = detector.detect(test_image)
        print(f"✅ Detection test passed (found {len(results['boxes'])} objects)")

    except Exception as e:
        print(f"❌ Detection model test failed: {e}")
        return False

    try:
        from segmentation import WaterSegmentation
        print("Loading segmentation model...")
        segmenter = WaterSegmentation('yolo11n-seg.pt', conf_threshold=0.25)
        print("✅ Segmentation model loaded successfully")

        # Test with dummy image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = segmenter.segment(test_image)
        print(f"✅ Segmentation test passed (found {len(results['masks'])} segments)")

    except Exception as e:
        print(f"❌ Segmentation model test failed: {e}")
        return False

    return True

def main():
    """Run all tests"""
    print("="*60)
    print("Vehicle Submersion Detection System - Setup Verification")
    print("="*60)
    print()

    # Run tests
    imports_ok = test_imports()
    modules_ok = test_modules()
    dirs_ok = test_directories()

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    if imports_ok and modules_ok and dirs_ok:
        print("✅ All basic tests passed!")
        print("\nYou can now run the application with:")
        print("  streamlit run app.py")
        print("or")
        print("  run.bat")

        # Ask if user wants to test model loading
        print("\n" + "="*60)
        response = input("\nDo you want to test model loading? (y/n): ")
        if response.lower() == 'y':
            models_ok = test_model_loading()
            if models_ok:
                print("\n✅ All tests passed! System is ready to use.")
            else:
                print("\n⚠️  Model loading failed. Check error messages above.")
        else:
            print("\nSkipping model loading test.")
            print("Models will be downloaded automatically when you run the app.")
    else:
        print("❌ Some tests failed. Please check error messages above.")
        print("\nMake sure to:")
        print("1. Activate your conda environment: conda activate yolov11")
        print("2. Install dependencies: pip install -r requirements.txt")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
