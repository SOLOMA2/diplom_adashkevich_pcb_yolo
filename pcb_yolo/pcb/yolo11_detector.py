from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    """Configuration parameters for YOLO11s detector."""
    
    # Model parameters
    weights_path: str = "yolo11s.pt"
    device: str = ""  # Empty string for auto-detection
    
    # Inference parameters
    img_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    augment: bool = False
    agnostic_nms: bool = False
    
    # Filtering parameters
    min_box_area: float = 100.0  # Minimum box area in pixels
    max_box_area_ratio: float = 0.35  # Maximum box area as ratio of image area
    min_confidence: float = 0.25  # Minimum confidence for decision making
    
    # Visualization parameters
    show_confidence: bool = True
    show_class_names: bool = True
    box_color: tuple[int, int, int] = (255, 64, 200)  # Magenta for green PCBs
    box_thickness: int = 3
    font_scale: float = 0.6
    
    # Processing parameters
    batch_size: int = 1
    num_workers: int = 4
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not Path(self.weights_path).exists():
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")
        if self.conf_threshold < 0 or self.conf_threshold > 1:
            raise ValueError("conf_threshold must be between 0 and 1")
        if self.iou_threshold < 0 or self.iou_threshold > 1:
            raise ValueError("iou_threshold must be between 0 and 1")
        if self.img_size <= 0:
            raise ValueError("img_size must be positive")


@dataclass
class DetectionResult:
    """Structured result from defect detection."""
    
    boxes: np.ndarray  # Shape: (N, 4) - [x1, y1, x2, y2]
    confidences: np.ndarray  # Shape: (N,) - confidence scores
    class_ids: np.ndarray  # Shape: (N,) - class indices
    class_names: list[str]  # Class names for each detection
    image_shape: tuple[int, int, int]  # Original image shape (H, W, C)
    inference_time_ms: float  # Inference time in milliseconds
    
    def has_defects(self, threshold: float = 0.25) -> bool:
        """Check if any detection exceeds confidence threshold."""
        return bool(np.any(self.confidences >= threshold))
    
    def get_defect_count(self) -> int:
        """Get total number of detected defects."""
        return len(self.confidences)
    
    def get_defects_by_class(self) -> dict[str, int]:
        """Get count of defects grouped by class name."""
        defect_counts = {}
        for class_name in self.class_names:
            defect_counts[class_name] = defect_counts.get(class_name, 0) + 1
        return defect_counts
    
    def get_max_confidence(self) -> float:
        """Get maximum confidence among all detections."""
        return float(np.max(self.confidences)) if len(self.confidences) > 0 else 0.0
    
    def get_average_confidence(self) -> float:
        """Get average confidence among all detections."""
        return float(np.mean(self.confidences)) if len(self.confidences) > 0 else 0.0


class YOLO11Detector:
    """
    High-level wrapper for YOLO11s defect detection with encapsulated logic.
    
    This class provides a clean interface for PCB defect detection, handling
    model loading, preprocessing, inference, postprocessing, and result filtering.
    """
    
    def __init__(self, config: DetectionConfig):
        """
        Initialize YOLO11s detector with configuration.
        
        Args:
            config: DetectionConfig object with all parameters
        """
        config.validate()
        self.config = config
        self.model: Optional[YOLO] = None
        self.class_names: list[str] = []
        self._load_model()
    
    def _load_model(self) -> None:
        """Load YOLO11s model and extract class names."""
        logger.info(f"Loading YOLO11s model from {self.config.weights_path}")
        
        try:
            self.model = YOLO(self.config.weights_path)
            
            # Extract and order class names
            raw_names = self.model.names
            if isinstance(raw_names, dict):
                self.class_names = [
                    raw_names[k] for k in sorted(
                        raw_names.keys(),
                        key=lambda x: int(x) if str(x).isdigit() else x
                    )
                ]
            else:
                self.class_names = list(raw_names)
            
            logger.info(f"Model loaded successfully. Classes: {self.class_names}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image in BGR format (OpenCV)
            
        Returns:
            Preprocessed image
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def filter_detections(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        image_shape: tuple[int, int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter detections based on confidence, area, and other criteria.
        
        Args:
            boxes: Detection boxes in xyxy format [N, 4]
            confidences: Confidence scores [N]
            class_ids: Class indices [N]
            image_shape: Image shape (H, W, C)
            
        Returns:
            Filtered boxes, confidences, and class_ids
        """
        if len(boxes) == 0:
            return boxes, confidences, class_ids
        
        h, w = image_shape[:2]
        max_area = self.config.max_box_area_ratio * (h * w)
        
        # Filter by confidence
        conf_mask = confidences >= self.config.min_confidence
        
        # Filter by area
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        area_mask = (areas >= self.config.min_box_area) & (areas <= max_area)
        
        # Combine filters
        combined_mask = conf_mask & area_mask
        
        return boxes[combined_mask], confidences[combined_mask], class_ids[combined_mask]
    
    def detect_single(
        self,
        image: np.ndarray,
        return_visualization: bool = False
    ) -> DetectionResult | tuple[DetectionResult, np.ndarray]:
        """
        Perform defect detection on a single image.
        
        Args:
            image: Input image in BGR format (OpenCV)
            return_visualization: If True, return visualization image
            
        Returns:
            DetectionResult (and optionally visualization image)
        """
        import time
        
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        original_shape = image.shape
        
        # Preprocess
        processed_image = self.preprocess_image(image)
        
        # Inference
        start_time = time.perf_counter()
        results = self.model(
            processed_image,
            imgsz=self.config.img_size,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            augment=self.config.augment,
            agnostic_nms=self.config.agnostic_nms,
            device=self.config.device,
            verbose=False
        )
        inference_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        # Extract results
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            detection_result = DetectionResult(
                boxes=np.zeros((0, 4), dtype=np.float32),
                confidences=np.zeros(0, dtype=np.float32),
                class_ids=np.zeros(0, dtype=np.int64),
                class_names=[],
                image_shape=original_shape,
                inference_time_ms=inference_time
            )
        else:
            boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            confidences = result.boxes.conf.cpu().numpy().astype(np.float32)
            class_ids = result.boxes.cls.cpu().numpy().astype(np.int64)
            
            # Filter detections
            boxes, confidences, class_ids = self.filter_detections(
                boxes, confidences, class_ids, original_shape
            )
            
            # Map class IDs to names
            class_names = [
                self.class_names[cid] if 0 <= cid < len(self.class_names) else str(cid)
                for cid in class_ids
            ]
            
            detection_result = DetectionResult(
                boxes=boxes,
                confidences=confidences,
                class_ids=class_ids,
                class_names=class_names,
                image_shape=original_shape,
                inference_time_ms=inference_time
            )
        
        if return_visualization:
            visualization = self.visualize_results(image, detection_result)
            return detection_result, visualization
        
        return detection_result
    
    def detect_batch(
        self,
        images: list[np.ndarray],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> list[DetectionResult]:
        """
        Perform defect detection on a batch of images.
        
        Args:
            images: List of input images in BGR format
            progress_callback: Optional callback for progress updates (current, total)
            
        Returns:
            List of DetectionResult objects
        """
        results = []
        total = len(images)
        
        for i, image in enumerate(images):
            result = self.detect_single(image)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results
    
    def visualize_results(
        self,
        image: np.ndarray,
        detection_result: DetectionResult
    ) -> np.ndarray:
        vis_image = image.copy()
        
        if detection_result.get_defect_count() == 0:
            return vis_image
        
        for box, conf, class_name in zip(
            detection_result.boxes,
            detection_result.confidences,
            detection_result.class_names
        ):
            x1, y1, x2, y2 = map(int, box)
            
            # Draw box
            cv2.rectangle(
                vis_image,
                (x1, y1),
                (x2, y2),
                self.config.box_color,
                self.config.box_thickness
            )
            
            # Draw label
            if self.config.show_class_names or self.config.show_confidence:
                label_parts = []
                if self.config.show_class_names:
                    label_parts.append(class_name)
                if self.config.show_confidence:
                    label_parts.append(f"{conf:.2f}")
                label = " ".join(label_parts)
                
                # Get label size
                (text_w, text_h), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.font_scale,
                    2
                )
                
                # Draw label background
                label_bg_y1 = max(0, y1 - text_h - baseline - 10)
                label_bg_y2 = y1
                cv2.rectangle(
                    vis_image,
                    (x1, label_bg_y1),
                    (x1 + text_w + 10, label_bg_y2),
                    self.config.box_color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    vis_image,
                    label,
                    (x1 + 5, label_bg_y2 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.font_scale,
                    (255, 255, 255),
                    2
                )
        
        return vis_image
    
    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {}
        
        return {
            "weights_path": self.config.weights_path,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "img_size": self.config.img_size,
            "device": str(self.model.device) if hasattr(self.model, 'device') else "unknown"
        }


def create_default_detector(weights_path: str = "yolo11s.pt") -> YOLO11Detector:

    config = DetectionConfig(weights_path=weights_path)
    return YOLO11Detector(config)


if __name__ == "__main__":
    # Example usage
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python yolo11_detector.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load image: {image_path}")
        sys.exit(1)
    
    # Create detector
    detector = create_default_detector()
    
    # Perform detection
    result, visualization = detector.detect_single(image, return_visualization=True)
    
    # Print results
    print(f"Detection completed in {result.inference_time_ms:.2f} ms")
    print(f"Defects found: {result.get_defect_count()}")
    print(f"Has defects: {result.has_defects()}")
    print(f"Max confidence: {result.get_max_confidence():.3f}")
    print(f"Defects by class: {result.get_defects_by_class()}")
    
    # Save visualization
    output_path = Path(image_path).stem + "_detected.jpg"
    cv2.imwrite(output_path, visualization)
    print(f"Visualization saved to: {output_path}")
