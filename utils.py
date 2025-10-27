"""
Utility functions for vehicle submersion analysis
支持不同淹没部位车辆的数量统计
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple


def calculate_submersion_stats(image, detection_results, segmentation_results):
    """
    Calculate vehicle submersion statistics
    统计不同淹没部位的车辆数量

    Handles two modes:
    1. Custom models: Detects vehicle parts (body, window, wheel) and analyzes part submersion
    2. COCO models: Detects whole vehicles and analyzes overall submersion

    Args:
        image (np.ndarray): Original image
        detection_results (dict): Vehicle detection results
        segmentation_results (dict): Water segmentation results

    Returns:
        dict: Statistics about vehicle submersion levels
    """
    if len(detection_results['boxes']) == 0:
        return {
            'total_vehicles': 0,
            'fully_submerged': 0,
            'partially_submerged': 0,
            'not_submerged': 0,
            'submersion_details': [],
            'is_custom_model': detection_results.get('is_custom_model', False)
        }

    # Get water mask
    water_mask = segmentation_results['combined_mask']
    is_custom_model = detection_results.get('is_custom_model', False)

    if is_custom_model:
        # Custom model: Analyze part-based submersion (body, window, wheel)
        return _calculate_part_based_stats(image, detection_results, water_mask)
    else:
        # COCO model: Analyze whole vehicle submersion
        return _calculate_vehicle_based_stats(image, detection_results, water_mask)


def _calculate_part_based_stats(image, detection_results, water_mask):
    """
    Calculate submersion stats for custom part-based detection
    Groups parts (body, window, wheel) and determines submersion level
    """
    submersion_details = []
    fully_submerged = 0
    partially_submerged = 0
    not_submerged = 0

    # Analyze each detected part
    for i, box in enumerate(detection_results['boxes']):
        x1, y1, x2, y2 = map(int, box)
        part_name = detection_results['class_names'][i]

        # Extract part region
        part_mask = np.zeros_like(water_mask)
        part_mask[y1:y2, x1:x2] = 1

        # Calculate overlap with water
        overlap = np.logical_and(part_mask, water_mask)
        overlap_pixels = np.sum(overlap)
        part_pixels = np.sum(part_mask)

        if part_pixels == 0:
            submersion_ratio = 0
        else:
            submersion_ratio = overlap_pixels / part_pixels

        # Determine submersion level based on PART TYPE and water coverage
        # Wheels underwater = partially submerged
        # Body underwater = fully submerged
        # Window underwater = fully submerged
        if part_name == 'wheel':
            if submersion_ratio >= 0.5:
                level = "partially_submerged"
                partially_submerged += 1
            else:
                level = "not_submerged"
                not_submerged += 1
        elif part_name in ['body', 'window']:
            if submersion_ratio >= 0.7:
                level = "fully_submerged"
                fully_submerged += 1
            elif submersion_ratio >= 0.3:
                level = "partially_submerged"
                partially_submerged += 1
            else:
                level = "not_submerged"
                not_submerged += 1
        else:
            # Unknown part
            if submersion_ratio >= 0.7:
                level = "fully_submerged"
                fully_submerged += 1
            elif submersion_ratio >= 0.2:
                level = "partially_submerged"
                partially_submerged += 1
            else:
                level = "not_submerged"
                not_submerged += 1

        submersion_details.append({
            'part_id': i,
            'part_type': part_name,
            'box': box.tolist(),
            'submersion_ratio': float(submersion_ratio),
            'level': level,
        })

    # Count unique vehicles (approximate: count body parts as vehicles)
    vehicle_count = sum(1 for detail in submersion_details if detail['part_type'] == 'body')
    if vehicle_count == 0:
        # No body detected, estimate from total parts
        vehicle_count = max(1, len(submersion_details) // 2)

    return {
        'total_vehicles': vehicle_count,
        'total_parts_detected': len(detection_results['boxes']),
        'fully_submerged': fully_submerged,
        'partially_submerged': partially_submerged,
        'not_submerged': not_submerged,
        'submersion_details': submersion_details,
        'is_custom_model': True
    }


def _calculate_vehicle_based_stats(image, detection_results, water_mask):
    """
    Calculate submersion stats for COCO whole-vehicle detection
    """
    submersion_details = []
    fully_submerged = 0
    partially_submerged = 0
    not_submerged = 0

    for i, box in enumerate(detection_results['boxes']):
        x1, y1, x2, y2 = map(int, box)

        # Extract vehicle region
        vehicle_mask = np.zeros_like(water_mask)
        vehicle_mask[y1:y2, x1:x2] = 1

        # Calculate overlap with water
        overlap = np.logical_and(vehicle_mask, water_mask)
        overlap_pixels = np.sum(overlap)
        vehicle_pixels = np.sum(vehicle_mask)

        if vehicle_pixels == 0:
            submersion_ratio = 0
        else:
            submersion_ratio = overlap_pixels / vehicle_pixels

        # Determine submersion level
        if submersion_ratio >= 0.7:
            level = "fully_submerged"
            fully_submerged += 1
        elif submersion_ratio >= 0.2:
            level = "partially_submerged"
            partially_submerged += 1
        else:
            level = "not_submerged"
            not_submerged += 1

        # Calculate which parts are submerged
        vehicle_height = y2 - y1
        submerged_parts = analyze_vehicle_parts(
            vehicle_mask[y1:y2, x1:x2],
            water_mask[y1:y2, x1:x2],
            vehicle_height
        )

        submersion_details.append({
            'vehicle_id': i,
            'class': detection_results['class_names'][i],
            'box': box.tolist(),
            'submersion_ratio': float(submersion_ratio),
            'level': level,
            'submerged_parts': submerged_parts
        })

    return {
        'total_vehicles': len(detection_results['boxes']),
        'fully_submerged': fully_submerged,
        'partially_submerged': partially_submerged,
        'not_submerged': not_submerged,
        'submersion_details': submersion_details,
        'is_custom_model': False
    }


def analyze_vehicle_parts(vehicle_mask, water_mask, vehicle_height):
    """
    Analyze which parts of the vehicle are submerged

    Args:
        vehicle_mask (np.ndarray): Binary mask of vehicle
        water_mask (np.ndarray): Binary mask of water
        vehicle_height (int): Height of the vehicle bounding box

    Returns:
        dict: Information about which parts are submerged
    """
    # Divide vehicle into parts (top, middle, bottom)
    third_height = vehicle_height // 3

    parts = {
        'top': False,
        'middle': False,
        'bottom': False,
        'wheels': False
    }

    if vehicle_height < 3:
        return parts

    # Check top third
    top_overlap = np.sum(np.logical_and(
        vehicle_mask[:third_height, :],
        water_mask[:third_height, :]
    ))
    top_pixels = np.sum(vehicle_mask[:third_height, :])
    if top_pixels > 0 and (top_overlap / top_pixels) > 0.3:
        parts['top'] = True

    # Check middle third
    middle_overlap = np.sum(np.logical_and(
        vehicle_mask[third_height:2*third_height, :],
        water_mask[third_height:2*third_height, :]
    ))
    middle_pixels = np.sum(vehicle_mask[third_height:2*third_height, :])
    if middle_pixels > 0 and (middle_overlap / middle_pixels) > 0.3:
        parts['middle'] = True

    # Check bottom third (wheels area)
    bottom_overlap = np.sum(np.logical_and(
        vehicle_mask[2*third_height:, :],
        water_mask[2*third_height:, :]
    ))
    bottom_pixels = np.sum(vehicle_mask[2*third_height:, :])
    if bottom_pixels > 0 and (bottom_overlap / bottom_pixels) > 0.3:
        parts['bottom'] = True
        parts['wheels'] = True

    return parts


def visualize_results(image, detection_results, segmentation_results,
                     show_detection=True, show_segmentation=True):
    """
    Visualize detection and segmentation results

    Args:
        image (np.ndarray): Original image
        detection_results (dict): Vehicle detection results
        segmentation_results (dict): Water segmentation results
        show_detection (bool): Whether to show detection boxes
        show_segmentation (bool): Whether to show segmentation masks

    Returns:
        np.ndarray: Visualized image
    """
    vis_image = image.copy()

    # Add segmentation visualization
    if show_segmentation and len(segmentation_results['masks']) > 0:
        # Create blue overlay for water
        water_overlay = np.zeros_like(image)
        combined_mask = segmentation_results['combined_mask']

        # Use bright blue color for water - MORE VISIBLE
        water_overlay[combined_mask > 0] = [0, 150, 255]  # Bright blue

        # Blend with original image - INCREASED VISIBILITY (60% water, 40% image)
        vis_image = cv2.addWeighted(vis_image, 0.4, water_overlay, 0.6, 0)

        # Draw thicker contours for better visibility
        contours, _ = cv2.findContours(
            combined_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis_image, contours, -1, (0, 255, 255), 3)  # Cyan, thicker

    # Add detection visualization
    if show_detection and len(detection_results['boxes']) > 0:
        water_mask = segmentation_results['combined_mask']

        for i, box in enumerate(detection_results['boxes']):
            x1, y1, x2, y2 = map(int, box)

            # Calculate submersion level for color coding
            vehicle_mask = np.zeros_like(water_mask)
            vehicle_mask[y1:y2, x1:x2] = 1

            overlap = np.logical_and(vehicle_mask, water_mask)
            overlap_pixels = np.sum(overlap)
            vehicle_pixels = np.sum(vehicle_mask)

            if vehicle_pixels == 0:
                submersion_ratio = 0
            else:
                submersion_ratio = overlap_pixels / vehicle_pixels

            # Color code based on submersion level
            if submersion_ratio >= 0.7:
                color = (255, 0, 0)  # Red - fully submerged
                level_text = "Fully Submerged"
            elif submersion_ratio >= 0.2:
                color = (255, 165, 0)  # Orange - partially submerged
                level_text = "Partially Submerged"
            else:
                color = (0, 255, 0)  # Green - not submerged
                level_text = "Not Submerged"

            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            class_name = detection_results['class_names'][i]
            score = detection_results['scores'][i]
            label = f"{class_name}: {score:.2f}"
            label2 = f"{level_text} ({submersion_ratio:.1%})"

            # Background for text
            label_size1, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_size2, _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            label_y = max(y1 - 10, label_size1[1] + label_size2[1] + 10)

            # Draw background rectangles
            cv2.rectangle(
                vis_image,
                (x1, label_y - label_size1[1] - label_size2[1] - 10),
                (x1 + max(label_size1[0], label_size2[0]), label_y + 5),
                color,
                -1
            )

            # Draw text
            cv2.putText(
                vis_image,
                label,
                (x1, label_y - label_size2[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            cv2.putText(
                vis_image,
                label2,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

    return vis_image


def create_split_view(original_image, result_image, vertical=False):
    """
    Create a split-screen view of original and result images

    Args:
        original_image (np.ndarray): Original image
        result_image (np.ndarray): Processed result image
        vertical (bool): If True, split vertically; otherwise horizontally

    Returns:
        np.ndarray: Combined split-view image
    """
    if vertical:
        # Stack vertically
        combined = np.vstack([original_image, result_image])
    else:
        # Stack horizontally
        combined = np.hstack([original_image, result_image])

    return combined


def add_legend(image, position='bottom-right'):
    """
    Add a legend to explain color coding

    Args:
        image (np.ndarray): Image to add legend to
        position (str): Position of legend ('bottom-right', 'bottom-left', etc.)

    Returns:
        np.ndarray: Image with legend
    """
    legend_image = image.copy()
    h, w = image.shape[:2]

    # Legend content
    legend_items = [
        ("Fully Submerged", (255, 0, 0)),
        ("Partially Submerged", (255, 165, 0)),
        ("Not Submerged", (0, 255, 0)),
        ("Water Area", (0, 150, 255))  # Updated to match new water color
    ]

    # Calculate legend size
    legend_height = len(legend_items) * 30 + 20
    legend_width = 200

    # Determine position
    if position == 'bottom-right':
        x_offset = w - legend_width - 10
        y_offset = h - legend_height - 10
    elif position == 'bottom-left':
        x_offset = 10
        y_offset = h - legend_height - 10
    elif position == 'top-right':
        x_offset = w - legend_width - 10
        y_offset = 10
    else:  # top-left
        x_offset = 10
        y_offset = 10

    # Draw legend background
    cv2.rectangle(
        legend_image,
        (x_offset, y_offset),
        (x_offset + legend_width, y_offset + legend_height),
        (255, 255, 255),
        -1
    )
    cv2.rectangle(
        legend_image,
        (x_offset, y_offset),
        (x_offset + legend_width, y_offset + legend_height),
        (0, 0, 0),
        2
    )

    # Draw legend items
    for i, (text, color) in enumerate(legend_items):
        y = y_offset + 25 + i * 30

        # Draw color box
        cv2.rectangle(
            legend_image,
            (x_offset + 10, y - 10),
            (x_offset + 30, y + 5),
            color,
            -1
        )
        cv2.rectangle(
            legend_image,
            (x_offset + 10, y - 10),
            (x_offset + 30, y + 5),
            (0, 0, 0),
            1
        )

        # Draw text
        cv2.putText(
            legend_image,
            text,
            (x_offset + 40, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1
        )

    return legend_image


def export_stats_to_csv(stats_list, output_path):
    """
    Export statistics to CSV file

    Args:
        stats_list (list): List of statistics dictionaries
        output_path (str): Path to save CSV file
    """
    import csv

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'image_name',
            'total_vehicles',
            'fully_submerged',
            'partially_submerged',
            'not_submerged'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for stats in stats_list:
            writer.writerow(stats)


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")

    # Create dummy data
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    detection_results = {
        'boxes': [np.array([100, 100, 200, 200])],
        'scores': [0.95],
        'class_ids': [2],
        'class_names': ['car']
    }

    segmentation_results = {
        'masks': [np.ones((640, 640), dtype=np.uint8)],
        'combined_mask': np.ones((640, 640), dtype=np.uint8)
    }

    # Test statistics calculation
    stats = calculate_submersion_stats(test_image, detection_results, segmentation_results)
    print(f"Stats: {stats}")

    print("✅ Utility functions test complete!")
