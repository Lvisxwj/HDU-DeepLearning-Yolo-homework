"""
水积识别和车辆淹没部位判别系统
Water Accumulation Identification and Vehicle Submersion Detection System
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import json
from datetime import datetime
import io
import zipfile

from detection import VehicleDetector
from segmentation import WaterSegmentation
from utils import calculate_submersion_stats, visualize_results, create_split_view
import config

# Page configuration
st.set_page_config(
    page_title="Vehicle Submersion Detection System",
    page_icon="🚗",
    layout="wide"
)

# Initialize session state
if 'detection_model' not in st.session_state:
    st.session_state.detection_model = None
if 'segmentation_model' not in st.session_state:
    st.session_state.segmentation_model = None
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []

# Create necessary directories
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
TEMP_DIR = Path("temp")

for dir_path in [UPLOAD_DIR, RESULTS_DIR, MODELS_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

def main():
    st.title("🚗 水积识别和车辆淹没部位判别系统")
    st.markdown("### Water Accumulation Identification and Vehicle Submersion Detection System")

    # Sidebar for model selection
    with st.sidebar:
        st.header("⚙️ Model Configuration")

        # Detection model selection
        st.subheader("🎯 Object Detection Model")
        detection_models = config.DETECTION_MODELS

        selected_detection = st.selectbox(
            "Select Detection Model",
            list(detection_models.keys()),
            index=0,
            help="Custom trained models are marked with 🎯"
        )

        # Segmentation model selection
        st.subheader("🌊 Semantic Segmentation Model")
        segmentation_models = config.SEGMENTATION_MODELS

        selected_segmentation = st.selectbox(
            "Select Segmentation Model",
            list(segmentation_models.keys()),
            index=0,
            help="Custom trained models are marked with 🌊"
        )

        # Detection confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.01,
            max_value=1.0,
            value=0.10,
            step=0.01,
            format="%.2f",
            help="Your custom model works best at 0.10-0.15. Lower for more detections, higher for fewer."
        )

        st.markdown("---")
        st.success("✅ Custom trained models loaded")

        st.warning("⚠️ **Important:** Your vehicle model was trained on different images than flood dataset. Detection may be poor on these images.")

        with st.expander("ℹ️ Why low detection?"):
            st.write("""
            **Your vehicle model:**
            - Trained for 100 epochs on project1 dataset
            - Expects: ground-level, clear vehicle photos

            **These test images (cc*.jpg):**
            - Aerial flood photos with submerged vehicles
            - Different angle, lighting, and conditions

            **Solutions:**
            1. Use images from your **training dataset** (project1)
            2. Or retrain model with flood images
            3. Or use official YOLO models (YOLOv11n) for general vehicles
            """)

        st.info("💡 For best results: Use images from your training dataset or switch to official YOLO models")

    # Main content area
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 Statistics", "📥 Download Results"])

    # Tab 1: Upload and Process
    with tab1:
        st.header("Image Upload and Processing")

        # Add clear results button
        if st.session_state.processed_results:
            if st.button("🗑️ Clear All Results", type="secondary"):
                st.session_state.processed_results = []
                st.success("✅ Results cleared! Upload new images to process.")
                st.rerun()

        # File upload
        uploaded_files = st.file_uploader(
            "Upload Images (支持图像的上传)",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.success(f"✅ Uploaded {len(uploaded_files)} image(s)")

            # Process button
            if st.button("🚀 Process Images", type="primary"):
                process_images(
                    uploaded_files,
                    detection_models[selected_detection],
                    segmentation_models[selected_segmentation],
                    confidence_threshold
                )

        # ALWAYS display results if they exist (outside button, so radio buttons work)
        if st.session_state.processed_results:
            st.markdown("---")
            st.markdown("## 📊 Processing Results")
            for result in st.session_state.processed_results:
                display_result(result)

    # Tab 2: Statistics
    with tab2:
        st.header("Vehicle Submersion Statistics")
        display_statistics()

    # Tab 3: Download Results
    with tab3:
        st.header("Download Detection and Segmentation Results")
        display_download_section()

def process_images(uploaded_files, detection_model_name, segmentation_model_name, conf_threshold):
    """Process uploaded images with detection and segmentation"""

    # Initialize models
    with st.spinner("Loading models..."):
        detector = VehicleDetector(detection_model_name, conf_threshold)
        segmenter = WaterSegmentation(segmentation_model_name, conf_threshold)

    st.session_state.processed_results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}... ({idx + 1}/{len(uploaded_files)})")

        # Read image and convert to RGB
        try:
            image = Image.open(uploaded_file)

            # Always convert to RGB for consistency
            if image.mode == 'RGBA':
                # Create white background for transparency
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3])
                image = rgb_image
            elif image.mode != 'RGB':
                # Convert any other mode (L, P, CMYK, etc.) to RGB
                image = image.convert('RGB')

            img_array = np.array(image)

            # Ensure array is in correct format (H, W, 3)
            if len(img_array.shape) == 2:
                # Grayscale - convert to RGB
                img_array = np.stack([img_array] * 3, axis=-1)

        except Exception as e:
            st.error(f"Error reading image {uploaded_file.name}: {e}")
            continue

        # Save original image (always as PNG to preserve quality)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = Path(uploaded_file.name).stem + ".png"
        original_path = UPLOAD_DIR / f"{timestamp}_{original_filename}"

        try:
            image.save(original_path, 'PNG')
        except Exception as e:
            st.warning(f"Could not save original image: {e}")

        # Run detection
        with st.spinner(f"Running vehicle detection on {uploaded_file.name}..."):
            detection_results = detector.detect(img_array)
            st.info(f"Detection: Found {len(detection_results['boxes'])} objects - Classes: {set(detection_results['class_names']) if detection_results['class_names'] else 'None'}")

        # Run segmentation
        with st.spinner(f"Running water segmentation on {uploaded_file.name}..."):
            segmentation_results = segmenter.segment(img_array)
            st.info(f"Segmentation: Found {len(segmentation_results['masks'])} segments - Classes: {set(segmentation_results['class_names']) if segmentation_results['class_names'] else 'None'}")

        # Calculate submersion statistics
        stats = calculate_submersion_stats(
            img_array,
            detection_results,
            segmentation_results
        )

        # Create visualizations
        detection_vis = visualize_results(
            img_array,
            detection_results,
            segmentation_results,
            show_detection=True,
            show_segmentation=False
        )

        segmentation_vis = visualize_results(
            img_array,
            detection_results,
            segmentation_results,
            show_detection=False,
            show_segmentation=True
        )

        combined_vis = visualize_results(
            img_array,
            detection_results,
            segmentation_results,
            show_detection=True,
            show_segmentation=True
        )

        # Save results
        result_prefix = RESULTS_DIR / f"{timestamp}_{Path(uploaded_file.name).stem}"

        cv2.imwrite(str(result_prefix) + "_detection.jpg", cv2.cvtColor(detection_vis, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(result_prefix) + "_segmentation.jpg", cv2.cvtColor(segmentation_vis, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(result_prefix) + "_combined.jpg", cv2.cvtColor(combined_vis, cv2.COLOR_RGB2BGR))

        # Save stats as JSON
        with open(str(result_prefix) + "_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        # Store results
        result_data = {
            "filename": uploaded_file.name,
            "timestamp": timestamp,
            "original_image": img_array,
            "detection_image": detection_vis,
            "segmentation_image": segmentation_vis,
            "combined_image": combined_vis,
            "stats": stats,
            "original_path": str(original_path),
            "result_prefix": str(result_prefix)
        }

        st.session_state.processed_results.append(result_data)

        progress_bar.progress((idx + 1) / len(uploaded_files))

    status_text.text("✅ Processing complete!")

    # Display results with split view
    st.success(f"Successfully processed {len(uploaded_files)} image(s)")

    for result in st.session_state.processed_results:
        display_result(result)

def display_result(result):
    """Display a single result with split-screen view"""
    st.markdown("---")
    st.subheader(f"📸 {result['filename']}")

    # Create split-screen view
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Image**")
        st.image(result['original_image'], use_container_width=True)

    with col2:
        view_option = st.radio(
            "Select View",
            ["Detection Only", "Segmentation Only", "Combined"],
            key=f"view_{result['timestamp']}",
            horizontal=True
        )

        if view_option == "Detection Only":
            st.markdown("**Detection Results**")
            st.image(result['detection_image'], use_container_width=True)
        elif view_option == "Segmentation Only":
            st.markdown("**Segmentation Results**")
            st.image(result['segmentation_image'], use_container_width=True)
        else:
            st.markdown("**Combined Results**")
            st.image(result['combined_image'], use_container_width=True)

    # Display statistics
    st.markdown("**📊 Detection Statistics**")
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

    stats = result['stats']
    with stats_col1:
        st.metric("Total Vehicles", stats['total_vehicles'])
    with stats_col2:
        st.metric("Fully Submerged", stats['fully_submerged'])
    with stats_col3:
        st.metric("Partially Submerged", stats['partially_submerged'])
    with stats_col4:
        st.metric("Not Submerged", stats['not_submerged'])

def display_statistics():
    """Display aggregate statistics across all processed images"""
    if not st.session_state.processed_results:
        st.info("👆 Please upload and process images first")
        return

    # Aggregate statistics
    total_vehicles = sum(r['stats']['total_vehicles'] for r in st.session_state.processed_results)
    fully_submerged = sum(r['stats']['fully_submerged'] for r in st.session_state.processed_results)
    partially_submerged = sum(r['stats']['partially_submerged'] for r in st.session_state.processed_results)
    not_submerged = sum(r['stats']['not_submerged'] for r in st.session_state.processed_results)

    # Display metrics
    st.markdown("### Overall Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Images Processed", len(st.session_state.processed_results))
    with col2:
        st.metric("Total Vehicles Detected", total_vehicles)
    with col3:
        st.metric("Total Fully Submerged", fully_submerged)
    with col4:
        st.metric("Total Partially Submerged", partially_submerged)

    # Create visualization
    st.markdown("### Submersion Level Distribution")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Bar chart data
        import pandas as pd
        data = {
            'Submersion Level': ['Not Submerged', 'Partially Submerged', 'Fully Submerged'],
            'Count': [not_submerged, partially_submerged, fully_submerged]
        }
        df = pd.DataFrame(data)
        st.bar_chart(df.set_index('Submersion Level'))

    with chart_col2:
        # Detailed table
        st.markdown("#### Per-Image Breakdown")
        table_data = []
        for result in st.session_state.processed_results:
            table_data.append({
                'Image': result['filename'],
                'Total': result['stats']['total_vehicles'],
                'Fully Submerged': result['stats']['fully_submerged'],
                'Partially Submerged': result['stats']['partially_submerged'],
                'Not Submerged': result['stats']['not_submerged']
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

def display_download_section():
    """Display download options for results"""
    if not st.session_state.processed_results:
        st.info("👆 Please upload and process images first")
        return

    st.markdown("### Available Downloads")

    # Option 1: Download all results as ZIP
    if st.button("📦 Download All Results as ZIP"):
        zip_buffer = create_results_zip()

        st.download_button(
            label="⬇️ Download ZIP File",
            data=zip_buffer,
            file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip"
        )

    st.markdown("---")

    # Option 2: Download individual results
    st.markdown("### Individual Downloads")

    for result in st.session_state.processed_results:
        with st.expander(f"📄 {result['filename']}"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                # Detection image
                det_buffer = io.BytesIO()
                Image.fromarray(result['detection_image']).save(det_buffer, format='PNG')
                st.download_button(
                    "Detection",
                    det_buffer.getvalue(),
                    f"{Path(result['filename']).stem}_detection.png",
                    "image/png",
                    key=f"det_{result['timestamp']}"
                )

            with col2:
                # Segmentation image
                seg_buffer = io.BytesIO()
                Image.fromarray(result['segmentation_image']).save(seg_buffer, format='PNG')
                st.download_button(
                    "Segmentation",
                    seg_buffer.getvalue(),
                    f"{Path(result['filename']).stem}_segmentation.png",
                    "image/png",
                    key=f"seg_{result['timestamp']}"
                )

            with col3:
                # Combined image
                comb_buffer = io.BytesIO()
                Image.fromarray(result['combined_image']).save(comb_buffer, format='PNG')
                st.download_button(
                    "Combined",
                    comb_buffer.getvalue(),
                    f"{Path(result['filename']).stem}_combined.png",
                    "image/png",
                    key=f"comb_{result['timestamp']}"
                )

            with col4:
                # Statistics JSON
                stats_json = json.dumps(result['stats'], indent=2)
                st.download_button(
                    "Statistics",
                    stats_json,
                    f"{Path(result['filename']).stem}_stats.json",
                    "application/json",
                    key=f"stats_{result['timestamp']}"
                )

def create_results_zip():
    """Create a ZIP file containing all results"""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for result in st.session_state.processed_results:
            prefix = Path(result['filename']).stem

            # Add images
            det_img = io.BytesIO()
            Image.fromarray(result['detection_image']).save(det_img, format='PNG')
            zip_file.writestr(f"{prefix}_detection.png", det_img.getvalue())

            seg_img = io.BytesIO()
            Image.fromarray(result['segmentation_image']).save(seg_img, format='PNG')
            zip_file.writestr(f"{prefix}_segmentation.png", seg_img.getvalue())

            comb_img = io.BytesIO()
            Image.fromarray(result['combined_image']).save(comb_img, format='PNG')
            zip_file.writestr(f"{prefix}_combined.png", comb_img.getvalue())

            # Add stats
            zip_file.writestr(f"{prefix}_stats.json", json.dumps(result['stats'], indent=2))

    zip_buffer.seek(0)
    return zip_buffer

if __name__ == "__main__":
    main()
