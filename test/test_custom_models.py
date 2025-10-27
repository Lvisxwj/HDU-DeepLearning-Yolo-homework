"""
Test script specifically for custom trained models
Run this to verify your custom models load and work correctly
"""

import sys
import numpy as np
from pathlib import Path
import io

# Set console encoding to UTF-8 for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def check_model_files():
    """Check if custom model files exist"""
    print("="*60)
    print("Checking Custom Model Files")
    print("="*60)

    models = [
        ("Vehicle Detection (Best)", "models/vehical/best.pt"),
        ("Vehicle Detection (Last)", "models/vehical/last.pt"),
        ("Flood Segmentation (Best)", "models/flood/best.pt"),
        ("Flood Segmentation (Last)", "models/flood/last.pt"),
    ]

    all_exist = True
    for name, path in models:
        if Path(path).exists():
            size = Path(path).stat().st_size / (1024 * 1024)  # Convert to MB
            print(f"✅ {name}: {path} ({size:.2f} MB)")
        else:
            print(f"❌ {name}: {path} NOT FOUND")
            all_exist = False

    return all_exist

def test_custom_detection_model():
    """Test custom vehicle detection model"""
    print("\n" + "="*60)
    print("Testing Custom Vehicle Detection Model")
    print("="*60)

    try:
        from detection import VehicleDetector

        print("\n📦 Loading Vehicle Custom (Best) model...")
        detector = VehicleDetector('models/vehical/best.pt', conf_threshold=0.25)
        print("✅ Model loaded successfully!")

        # Create test image
        print("🖼️  Creating test image...")
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Run detection
        print("🔍 Running detection...")
        results = detector.detect(test_image)

        print(f"✅ Detection completed!")
        print(f"   - Detected objects: {len(results['boxes'])}")
        print(f"   - Vehicle count: {detector.get_vehicle_count(results)}")

        if len(results['boxes']) > 0:
            print(f"   - Classes found: {set(results['class_names'])}")
        else:
            print("   - No objects detected (normal for random test image)")

        return True

    except Exception as e:
        print(f"❌ Error testing vehicle detection model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_custom_segmentation_model():
    """Test custom flood segmentation model"""
    print("\n" + "="*60)
    print("Testing Custom Flood Segmentation Model")
    print("="*60)

    try:
        from segmentation import WaterSegmentation

        print("\n📦 Loading Flood Custom (Best) model...")
        segmenter = WaterSegmentation('models/flood/best.pt', conf_threshold=0.25)
        print("✅ Model loaded successfully!")

        # Create test image
        print("🖼️  Creating test image...")
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Run segmentation
        print("🔍 Running segmentation...")
        results = segmenter.segment(test_image)

        print(f"✅ Segmentation completed!")
        print(f"   - Segments found: {len(results['masks'])}")
        print(f"   - Combined mask size: {results['combined_mask'].shape}")

        if len(results['masks']) > 0:
            print(f"   - Classes found: {set(results['class_names'])}")
            stats = segmenter.get_segmentation_stats(results)
            print(f"   - Coverage: {stats['coverage_percent']:.2f}%")
        else:
            print("   - No segments detected (normal for random test image)")

        return True

    except Exception as e:
        print(f"❌ Error testing flood segmentation model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_both_models_together():
    """Test detection and segmentation together (integration test)"""
    print("\n" + "="*60)
    print("Integration Test: Both Models Together")
    print("="*60)

    try:
        from detection import VehicleDetector
        from segmentation import WaterSegmentation
        from utils import calculate_submersion_stats, visualize_results

        print("\n📦 Loading both models...")
        detector = VehicleDetector('models/vehical/best.pt', conf_threshold=0.25)
        segmenter = WaterSegmentation('models/flood/best.pt', conf_threshold=0.25)
        print("✅ Both models loaded!")

        # Create test image
        print("🖼️  Creating test image...")
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Run both
        print("🔍 Running detection and segmentation...")
        detection_results = detector.detect(test_image)
        segmentation_results = segmenter.segment(test_image)

        print("📊 Calculating submersion statistics...")
        stats = calculate_submersion_stats(
            test_image,
            detection_results,
            segmentation_results
        )

        print(f"✅ Integration test passed!")
        print(f"   - Total vehicles: {stats['total_vehicles']}")
        print(f"   - Fully submerged: {stats['fully_submerged']}")
        print(f"   - Partially submerged: {stats['partially_submerged']}")
        print(f"   - Not submerged: {stats['not_submerged']}")

        # Test visualization
        print("🎨 Testing visualization...")
        vis_image = visualize_results(
            test_image,
            detection_results,
            segmentation_results,
            show_detection=True,
            show_segmentation=True
        )
        print(f"✅ Visualization created! Shape: {vis_image.shape}")

        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_switching():
    """Test switching between best and last models"""
    print("\n" + "="*60)
    print("Testing Model Switching")
    print("="*60)

    try:
        from detection import VehicleDetector

        print("\n📦 Testing vehicle model switching...")

        # Load best model
        print("1️⃣  Loading 'best' model...")
        detector = VehicleDetector('models/vehical/best.pt', conf_threshold=0.25)
        print("✅ Loaded: best.pt")

        # Switch to last model
        print("2️⃣  Switching to 'last' model...")
        detector.switch_model('models/vehical/last.pt')
        print("✅ Switched to: last.pt")

        # Test detection with switched model
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = detector.detect(test_image)
        print(f"✅ Detection with switched model works! ({len(results['boxes'])} objects)")

        return True

    except Exception as e:
        print(f"❌ Model switching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_best_vs_last():
    """Compare best.pt vs last.pt models"""
    print("\n" + "="*60)
    print("Comparing Best vs Last Models")
    print("="*60)

    try:
        from detection import VehicleDetector

        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        print("\n🔵 Testing Vehicle Best Model...")
        detector_best = VehicleDetector('models/vehical/best.pt', conf_threshold=0.25)
        results_best = detector_best.detect(test_image)
        print(f"   Detections: {len(results_best['boxes'])}")

        print("\n🟢 Testing Vehicle Last Model...")
        detector_last = VehicleDetector('models/vehical/last.pt', conf_threshold=0.25)
        results_last = detector_last.detect(test_image)
        print(f"   Detections: {len(results_last['boxes'])}")

        print("\n📊 Comparison (on random test image):")
        print(f"   Best model: {len(results_best['boxes'])} detections")
        print(f"   Last model: {len(results_last['boxes'])} detections")
        print(f"   Note: Results may vary with real images")

        return True

    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False

def main():
    """Run all custom model tests"""
    print("\n")
    print("╔" + "═"*58 + "╗")
    print("║" + " "*8 + "CUSTOM MODEL TESTING SUITE" + " "*24 + "║")
    print("╚" + "═"*58 + "╝")

    # Test 1: Check files exist
    files_ok = check_model_files()

    if not files_ok:
        print("\n" + "="*60)
        print("❌ CRITICAL: Model files not found!")
        print("="*60)
        print("\nPlease ensure your model files are in:")
        print("  - models/vehical/best.pt")
        print("  - models/vehical/last.pt")
        print("  - models/flood/best.pt")
        print("  - models/flood/last.pt")
        return 1

    # Test 2: Test detection model
    detection_ok = test_custom_detection_model()

    # Test 3: Test segmentation model
    segmentation_ok = test_custom_segmentation_model()

    # Test 4: Test both together
    integration_ok = test_both_models_together()

    # Test 5: Test model switching
    switching_ok = test_model_switching()

    # Test 6: Compare best vs last
    comparison_ok = compare_best_vs_last()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    tests = [
        ("Model Files Check", files_ok),
        ("Vehicle Detection Model", detection_ok),
        ("Flood Segmentation Model", segmentation_ok),
        ("Integration Test", integration_ok),
        ("Model Switching", switching_ok),
        ("Best vs Last Comparison", comparison_ok),
    ]

    passed = sum(1 for _, result in tests if result)
    total = len(tests)

    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)

    if passed == total:
        print("\n🎉 All tests passed! Your custom models are ready to use!")
        print("\nRecommended configuration for your demo:")
        print("  🎯 Detection Model: Vehicle Custom (Best)")
        print("  🌊 Segmentation Model: Flood Custom (Best)")
        print("\nRun the application:")
        print("  streamlit run app.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check error messages above.")
        print("\nTroubleshooting:")
        print("  1. Ensure models are in correct directories")
        print("  2. Check model compatibility with current Ultralytics version")
        print("  3. Try with real images instead of random test images")
        return 1

if __name__ == "__main__":
    sys.exit(main())
