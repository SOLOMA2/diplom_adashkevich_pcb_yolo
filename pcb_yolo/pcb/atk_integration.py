from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional
import uuid

import numpy as np
import cv2

from pcb.yolo11_detector import DetectionResult, YOLO11Detector


logger = logging.getLogger(__name__)


class InspectionVerdict(Enum):
    """Enumeration of possible inspection verdicts."""
    OK = "OK"  # No defects found
    NG = "NG"  # Defects found (No Good)
    RECHECK = "RECHECK"  # Uncertain result, requires manual inspection
    ERROR = "ERROR"  # Processing error occurred


@dataclass
class InspectionReport:
    
    # Identification
    inspection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    board_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Detection results
    verdict: InspectionVerdict = InspectionVerdict.OK
    defect_count: int = 0
    defects_by_class: dict[str, int] = field(default_factory=dict)
    max_confidence: float = 0.0
    avg_confidence: float = 0.0
    
    # Performance metrics
    inference_time_ms: float = 0.0
    total_processing_time_ms: float = 0.0
    
    # Image data
    image_path: str = ""
    result_image_path: str = ""
    
    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "inspection_id": self.inspection_id,
            "board_id": self.board_id,
            "timestamp": self.timestamp,
            "verdict": self.verdict.value,
            "defect_count": self.defect_count,
            "defects_by_class": self.defects_by_class,
            "max_confidence": self.max_confidence,
            "avg_confidence": self.avg_confidence,
            "inference_time_ms": self.inference_time_ms,
            "total_processing_time_ms": self.total_processing_time_ms,
            "image_path": self.image_path,
            "result_image_path": self.result_image_path,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def save_to_file(self, filepath: Path) -> None:
        """Save report to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
        logger.info(f"Report saved to {filepath}")


class DecisionEngine:
    
    def __init__(
        self,
        confidence_threshold: float = 0.25,
        max_defect_count: int = 0,
        class_rules: Optional[dict[str, dict[str, Any]]] = None
    ):
        """
        Initialize decision engine.
        
        Args:
            confidence_threshold: Minimum confidence for defect consideration
            max_defect_count: Maximum allowed defects before NG verdict
            class_rules: Class-specific rules (e.g., {"scratch": {"max_count": 5}})
        """
        self.confidence_threshold = confidence_threshold
        self.max_defect_count = max_defect_count
        self.class_rules = class_rules or {}
        self.custom_decision_fn: Optional[Callable[[DetectionResult], InspectionVerdict]] = None
    
    def set_custom_decision_function(self, fn: Callable[[DetectionResult], InspectionVerdict]) -> None:
        """Set custom decision function for verdict determination."""
        self.custom_decision_fn = fn
    
    def determine_verdict(self, result: DetectionResult) -> InspectionVerdict:
        """
        Determine inspection verdict based on detection result.
        
        Args:
            result: DetectionResult from detector
            
        Returns:
            InspectionVerdict (OK, NG, RECHECK, or ERROR)
        """
        # Use custom decision function if provided
        if self.custom_decision_fn:
            return self.custom_decision_fn(result)
        
        # Check for processing errors
        if result.inference_time_ms < 0:
            return InspectionVerdict.ERROR
        
        # Check defect count
        defect_count = result.get_defect_count()
        
        if defect_count == 0:
            return InspectionVerdict.OK
        
        # Check against max defect count
        if defect_count > self.max_defect_count:
            return InspectionVerdict.NG
        
        # Check class-specific rules
        defects_by_class = result.get_defects_by_class()
        for class_name, count in defects_by_class.items():
            if class_name in self.class_rules:
                class_rule = self.class_rules[class_name]
                max_allowed = class_rule.get("max_count", self.max_defect_count)
                if count > max_allowed:
                    return InspectionVerdict.NG
        
        # Check confidence threshold
        if result.has_defects(self.confidence_threshold):
            return InspectionVerdict.NG
        
        # If defects exist but below threshold, mark for recheck
        return InspectionVerdict.RECHECK


class ATKIntegration:
    """
    Main integration class for connecting YOLO11s detector with ATK systems.
    
    Provides high-level interface for:
    - Running inspections
    - Generating reports
    - Logging results
    - Communicating with external systems
    """
    
    def __init__(
        self,
        detector: YOLO11Detector,
        decision_engine: DecisionEngine,
        output_dir: Path = Path("reports")
    ):
        """
        Initialize ATK integration.
        
        Args:
            detector: Configured YOLO11Detector instance
            decision_engine: DecisionEngine for verdict determination
            output_dir: Directory for saving reports and visualizations
        """
        self.detector = detector
        self.decision_engine = decision_engine
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.total_inspections = 0
        self.ok_count = 0
        self.ng_count = 0
        self.recheck_count = 0
        self.error_count = 0
        
        # Callbacks
        self.on_inspection_complete: Optional[Callable[[InspectionReport], None]] = None
        self.on_verdict_change: Optional[Callable[[InspectionVerdict], None]] = None
    
    def inspect_image(
        self,
        image: np.ndarray,
        board_id: str = "",
        save_visualization: bool = True,
        save_report: bool = True
    ) -> InspectionReport:
        """
        Perform complete inspection on a single image.
        
        Args:
            image: Input image in BGR format
            board_id: Optional board identifier
            save_visualization: Whether to save visualization image
            save_report: Whether to save JSON report
            
        Returns:
            InspectionReport with complete inspection data
        """
        import time
        
        start_time = time.perf_counter()
        
        # Perform detection
        detection_result, visualization = self.detector.detect_single(
            image, return_visualization=True
        )
        
        # Determine verdict
        verdict = self.decision_engine.determine_verdict(detection_result)
        
        # Create report
        report = InspectionReport(
            board_id=board_id,
            verdict=verdict,
            defect_count=detection_result.get_defect_count(),
            defects_by_class=detection_result.get_defects_by_class(),
            max_confidence=detection_result.get_max_confidence(),
            avg_confidence=detection_result.get_average_confidence(),
            inference_time_ms=detection_result.inference_time_ms,
            total_processing_time_ms=(time.perf_counter() - start_time) * 1000
        )
        
        # Save visualization if requested
        if save_visualization:
            vis_filename = f"{report.inspection_id}_vis.jpg"
            vis_path = self.output_dir / vis_filename
            cv2.imwrite(str(vis_path), visualization)
            report.result_image_path = str(vis_path)
        
        # Save report if requested
        if save_report:
            report_filename = f"{report.inspection_id}_report.json"
            report_path = self.output_dir / report_filename
            report.save_to_file(report_path)
        
        # Update statistics
        self.total_inspections += 1
        if verdict == InspectionVerdict.OK:
            self.ok_count += 1
        elif verdict == InspectionVerdict.NG:
            self.ng_count += 1
        elif verdict == InspectionVerdict.RECHECK:
            self.recheck_count += 1
        else:
            self.error_count += 1
        
        # Trigger callbacks
        if self.on_inspection_complete:
            self.on_inspection_complete(report)
        
        logger.info(
            f"Inspection {report.inspection_id}: {verdict.value}, "
            f"{report.defect_count} defects, {report.inference_time_ms:.2f} ms"
        )
        
        return report
    
    def inspect_batch(
        self,
        images: list[np.ndarray],
        board_ids: Optional[list[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> list[InspectionReport]:
        """
        Perform inspection on a batch of images.
        
        Args:
            images: List of input images
            board_ids: Optional list of board identifiers
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of InspectionReport objects
        """
        if board_ids is None:
            board_ids = [f"board_{i}" for i in range(len(images))]
        
        reports = []
        total = len(images)
        
        for i, (image, board_id) in enumerate(zip(images, board_ids)):
            report = self.inspect_image(image, board_id=board_id)
            reports.append(report)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return reports
    
    def get_statistics(self) -> dict[str, Any]:
        """Get inspection statistics."""
        total = self.total_inspections
        if total == 0:
            return {
                "total_inspections": 0,
                "ok_rate": 0.0,
                "ng_rate": 0.0,
                "recheck_rate": 0.0,
                "error_rate": 0.0
            }
        
        return {
            "total_inspections": total,
            "ok_count": self.ok_count,
            "ng_count": self.ng_count,
            "recheck_count": self.recheck_count,
            "error_count": self.error_count,
            "ok_rate": self.ok_count / total,
            "ng_rate": self.ng_count / total,
            "recheck_rate": self.recheck_count / total,
            "error_rate": self.error_count / total
        }
    
    def reset_statistics(self) -> None:
        """Reset inspection statistics."""
        self.total_inspections = 0
        self.ok_count = 0
        self.ng_count = 0
        self.recheck_count = 0
        self.error_count = 0


def create_atk_system(
    weights_path: str = "yolo11s.pt",
    confidence_threshold: float = 0.25,
    output_dir: Path = Path("reports")
) -> ATKIntegration:
    """
    Factory function to create a complete ATK inspection system.
    
    Args:
        weights_path: Path to YOLO11s weights
        confidence_threshold: Confidence threshold for decision making
        output_dir: Directory for reports and visualizations
        
    Returns:
        Configured ATKIntegration instance
    """
    from pcb.yolo11_detector import DetectionConfig, create_default_detector
    
    # Create detector
    detector = create_default_detector(weights_path)
    
    # Create decision engine
    decision_engine = DecisionEngine(confidence_threshold=confidence_threshold)
    
    # Create ATK integration
    atk_system = ATKIntegration(detector, decision_engine, output_dir)
    
    return atk_system


if __name__ == "__main__":
    # Example usage
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python atk_integration.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load image: {image_path}")
        sys.exit(1)
    
    # Create ATK system
    atk_system = create_atk_system()
    
    # Perform inspection
    report = atk_system.inspect_image(image, board_id="test_board_001")
    
    # Print results
    print(f"\nInspection Report:")
    print(f"ID: {report.inspection_id}")
    print(f"Board ID: {report.board_id}")
    print(f"Verdict: {report.verdict.value}")
    print(f"Defects: {report.defect_count}")
    print(f"Max Confidence: {report.max_confidence:.3f}")
    print(f"Inference Time: {report.inference_time_ms:.2f} ms")
    print(f"Total Time: {report.total_processing_time_ms:.2f} ms")
    
    # Print statistics
    stats = atk_system.get_statistics()
    print(f"\nStatistics:")
    print(f"Total Inspections: {stats['total_inspections']}")
    print(f"OK Rate: {stats['ok_rate']:.2%}")
    print(f"NG Rate: {stats['ng_rate']:.2%}")
