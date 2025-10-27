"""
Vehicle Detection Module using YOLO
支持选择和调用不同的目标检测模型
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path


class VehicleDetector:
    """Vehicle detection using YOLO models"""

    # Vehicle-related class IDs in COCO dataset (for official models)
    COCO_VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }

    # Custom model class IDs (for your trained models)
    CUSTOM_VEHICLE_PARTS = {
        0: 'body',
        1: 'window',
        2: 'wheel'
    }

    def __init__(self, model_name='yolo11n.pt', conf_threshold=0.25):
        """
        Initialize the vehicle detector

        Args:
            model_name (str): Name of the YOLO model to use
            conf_threshold (float): Confidence threshold for detection
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.model = None
        self.is_custom_model = False
        self.model_classes = {}
        self._load_model()

    def _load_model(self):
        """Load the YOLO model"""
        try:
            # YOLO will automatically download the model if not present
            self.model = YOLO(self.model_name)

            # Detect if this is a custom model by checking class names
            if hasattr(self.model, 'names'):
                self.model_classes = self.model.names
                # Check if it's our custom vehicle parts model
                if 'body' in str(self.model_classes.values()) and 'wheel' in str(self.model_classes.values()):
                    self.is_custom_model = True
                    print(f"✅ Loaded CUSTOM vehicle parts model: {self.model_name}")
                    print(f"   Classes: {list(self.model_classes.values())}")
                else:
                    self.is_custom_model = False
                    print(f"✅ Loaded detection model: {self.model_name}")
                    print(f"   Using COCO classes")

        except Exception as e:
            print(f"❌ Error loading model {self.model_name}: {e}")
            raise

    def detect(self, image):
        """
        Detect vehicles in an image

        Args:
            image (np.ndarray): Input image in RGB format

        Returns:
            dict: Detection results containing boxes, scores, and class IDs
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        # Run inference
        results = self.model(image, conf=self.conf_threshold, verbose=False)

        # Parse results
        detections = self._parse_results(results[0])

        return detections

    def _parse_results(self, result):
        """
        Parse YOLO results
        - For custom models: Return all detections (body, window, wheel)
        - For COCO models: Filter for vehicle classes only (car, truck, bus)

        Args:
            result: YOLO result object

        Returns:
            dict: Parsed detections
        """
        detections = {
            'boxes': [],
            'scores': [],
            'class_ids': [],
            'class_names': [],
            'is_custom_model': self.is_custom_model
        }

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        # Extract boxes, scores, and classes
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        if self.is_custom_model:
            # Custom model: Use all detections (body, window, wheel)
            for i, class_id in enumerate(class_ids):
                detections['boxes'].append(boxes[i])
                detections['scores'].append(scores[i])
                detections['class_ids'].append(class_id)
                # Get class name from model
                class_name = self.model_classes.get(class_id, f'class_{class_id}')
                detections['class_names'].append(class_name)
        else:
            # COCO model: Filter for vehicle classes only
            for i, class_id in enumerate(class_ids):
                if class_id in self.COCO_VEHICLE_CLASSES:
                    detections['boxes'].append(boxes[i])
                    detections['scores'].append(scores[i])
                    detections['class_ids'].append(class_id)
                    detections['class_names'].append(self.COCO_VEHICLE_CLASSES[class_id])

        return detections

    def visualize(self, image, detections, color=(0, 255, 0), thickness=2):
        """
        Visualize detection results on the image

        Args:
            image (np.ndarray): Input image
            detections (dict): Detection results
            color (tuple): Color for bounding boxes
            thickness (int): Thickness of bounding boxes

        Returns:
            np.ndarray: Image with visualized detections
        """
        vis_image = image.copy()

        for i, box in enumerate(detections['boxes']):
            x1, y1, x2, y2 = map(int, box)
            class_name = detections['class_names'][i]
            score = detections['scores'][i]

            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            label = f"{class_name}: {score:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1 - 10, label_size[1])

            cv2.rectangle(
                vis_image,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0], label_y + 5),
                color,
                -1
            )
            cv2.putText(
                vis_image,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        return vis_image

    def get_vehicle_count(self, detections):
        """
        Get count of detected vehicles

        Args:
            detections (dict): Detection results

        Returns:
            int: Number of detected vehicles
        """
        return len(detections['boxes'])

    def get_vehicle_by_type(self, detections):
        """
        Get vehicle count grouped by type

        Args:
            detections (dict): Detection results

        Returns:
            dict: Vehicle counts by type
        """
        counts = {}
        for class_name in detections['class_names']:
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts

    def switch_model(self, new_model_name):
        """
        Switch to a different detection model

        Args:
            new_model_name (str): Name of the new model
        """
        self.model_name = new_model_name
        self._load_model()
        print(f"✅ Switched to model: {new_model_name}")


if __name__ == "__main__":
    # Test the detector
    print("Testing Vehicle Detector...")

    detector = VehicleDetector('yolo11n.pt', conf_threshold=0.25)

    # Create a dummy image for testing
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Run detection
    results = detector.detect(test_image)
    print(f"Detected {detector.get_vehicle_count(results)} vehicles")
    print(f"Vehicle types: {detector.get_vehicle_by_type(results)}")

    print("✅ Detection module test complete!")
