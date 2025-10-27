"""
Water Segmentation Module using YOLO Segmentation
支持选择和调用不同的语义分割模型
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path


class WaterSegmentation:
    """Water and flood segmentation using YOLO segmentation models"""

    def __init__(self, model_name='yolo11n-seg.pt', conf_threshold=0.25):
        """
        Initialize the segmentation model

        Args:
            model_name (str): Name of the YOLO segmentation model
            conf_threshold (float): Confidence threshold for segmentation
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.model = None
        self.is_custom_model = False
        self.model_classes = {}
        self._load_model()

    def _load_model(self):
        """Load the YOLO segmentation model"""
        try:
            # YOLO will automatically download the model if not present
            self.model = YOLO(self.model_name)

            # Detect if this is a custom water segmentation model
            if hasattr(self.model, 'names'):
                self.model_classes = self.model.names
                # Check if it's our custom water model
                if 'water' in str(self.model_classes.values()) and len(self.model_classes) <= 2:
                    self.is_custom_model = True
                    print(f"✅ Loaded CUSTOM water segmentation model: {self.model_name}")
                    print(f"   Classes: {list(self.model_classes.values())}")
                else:
                    self.is_custom_model = False
                    print(f"✅ Loaded segmentation model: {self.model_name}")
                    print(f"   Total classes: {len(self.model_classes)}")

        except Exception as e:
            print(f"❌ Error loading model {self.model_name}: {e}")
            raise

    def segment(self, image):
        """
        Perform semantic segmentation on an image

        Args:
            image (np.ndarray): Input image in RGB format

        Returns:
            dict: Segmentation results containing masks and class information
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        # Run inference
        results = self.model(image, conf=self.conf_threshold, verbose=False)

        # Parse results
        segmentation = self._parse_results(results[0], image.shape[:2])

        return segmentation

    def _parse_results(self, result, image_shape):
        """
        Parse YOLO segmentation results

        Args:
            result: YOLO result object
            image_shape: Tuple of (height, width)

        Returns:
            dict: Parsed segmentation results
        """
        segmentation = {
            'masks': [],
            'boxes': [],
            'scores': [],
            'class_ids': [],
            'class_names': [],
            'combined_mask': np.zeros(image_shape, dtype=np.uint8),
            'is_custom_model': self.is_custom_model
        }

        if result.masks is None or len(result.masks) == 0:
            return segmentation

        # Extract masks and related information
        masks = result.masks.data.cpu().numpy()  # [N, H, W]
        boxes = result.boxes.xyxy.cpu().numpy()  # [N, 4]
        scores = result.boxes.conf.cpu().numpy()  # [N]
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # [N]

        # Get class names
        class_names = [result.names[class_id] for class_id in class_ids]

        # Resize masks to original image size
        for i in range(len(masks)):
            mask = masks[i]
            mask_resized = cv2.resize(
                mask,
                (image_shape[1], image_shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            mask_binary = (mask_resized > 0.5).astype(np.uint8)

            segmentation['masks'].append(mask_binary)
            segmentation['boxes'].append(boxes[i])
            segmentation['scores'].append(scores[i])
            segmentation['class_ids'].append(class_ids[i])
            segmentation['class_names'].append(class_names[i])

            # Combine masks
            segmentation['combined_mask'] = np.maximum(
                segmentation['combined_mask'],
                mask_binary
            )

        return segmentation

    def visualize(self, image, segmentation, alpha=0.5):
        """
        Visualize segmentation results on the image

        Args:
            image (np.ndarray): Input image
            segmentation (dict): Segmentation results
            alpha (float): Transparency for overlay

        Returns:
            np.ndarray: Image with visualized segmentation
        """
        vis_image = image.copy()

        # Create color map for different classes
        np.random.seed(42)
        colors = {}

        for i, class_name in enumerate(segmentation['class_names']):
            if class_name not in colors:
                colors[class_name] = tuple(np.random.randint(0, 255, 3).tolist())

            mask = segmentation['masks'][i]

            # Create colored mask
            color_mask = np.zeros_like(image)
            color = colors[class_name]
            color_mask[mask > 0] = color

            # Blend with original image
            vis_image = cv2.addWeighted(vis_image, 1, color_mask, alpha, 0)

            # Draw contours
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis_image, contours, -1, color, 2)

        return vis_image

    def get_water_mask(self, segmentation):
        """
        Extract water-related segmentation mask
        This assumes water/flood will be detected by the model

        Args:
            segmentation (dict): Segmentation results

        Returns:
            np.ndarray: Binary mask of water regions
        """
        # Keywords that might indicate water in segmentation
        water_keywords = ['water', 'flood', 'river', 'sea', 'ocean', 'lake']

        water_mask = np.zeros_like(segmentation['combined_mask'])

        for i, class_name in enumerate(segmentation['class_names']):
            # Check if class name contains water-related keywords
            if any(keyword in class_name.lower() for keyword in water_keywords):
                water_mask = np.maximum(water_mask, segmentation['masks'][i])

        # If no specific water class found, use combined mask as approximation
        if water_mask.sum() == 0:
            water_mask = segmentation['combined_mask']

        return water_mask

    def get_segmentation_stats(self, segmentation):
        """
        Get statistics about the segmentation

        Args:
            segmentation (dict): Segmentation results

        Returns:
            dict: Segmentation statistics
        """
        total_pixels = segmentation['combined_mask'].size
        segmented_pixels = np.sum(segmentation['combined_mask'] > 0)
        coverage_percent = (segmented_pixels / total_pixels) * 100

        class_coverage = {}
        for i, class_name in enumerate(segmentation['class_names']):
            mask_pixels = np.sum(segmentation['masks'][i] > 0)
            class_coverage[class_name] = (mask_pixels / total_pixels) * 100

        return {
            'total_segments': len(segmentation['masks']),
            'coverage_percent': coverage_percent,
            'class_coverage': class_coverage
        }

    def switch_model(self, new_model_name):
        """
        Switch to a different segmentation model

        Args:
            new_model_name (str): Name of the new model
        """
        self.model_name = new_model_name
        self._load_model()
        print(f"✅ Switched to model: {new_model_name}")


def create_water_level_mask(image_height, water_percentage=0.3):
    """
    Create a simulated water level mask for testing
    This creates a mask where the bottom portion of the image is marked as water

    Args:
        image_height (int): Height of the image
        water_percentage (float): Percentage of image height covered by water

    Returns:
        np.ndarray: Binary mask representing water level
    """
    mask = np.zeros((image_height, 1), dtype=np.uint8)
    water_level = int(image_height * (1 - water_percentage))
    mask[water_level:] = 1
    return mask


if __name__ == "__main__":
    # Test the segmentation module
    print("Testing Water Segmentation...")

    segmenter = WaterSegmentation('yolo11n-seg.pt', conf_threshold=0.25)

    # Create a dummy image for testing
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Run segmentation
    results = segmenter.segment(test_image)
    print(f"Segmentation complete")
    print(f"Number of segments: {len(results['masks'])}")

    stats = segmenter.get_segmentation_stats(results)
    print(f"Stats: {stats}")

    print("✅ Segmentation module test complete!")
