from typing import List, Sequence, Tuple

import numpy as np

from sceneflow.runners._factory import load_text_detector
from sceneflow.runners._helpers import Detection
from sceneflow.utils.logger import logger


class OCRProcessor:
    def __init__(self, detectors: List, *, device: str) -> None:
        self.detectors = detectors
        self.device = device

    @classmethod
    def from_pretrained(cls, detectors: Sequence[str], *, device: str = "cuda") -> "OCRProcessor":
        """Instantiate an OCRRunner with a list of detector names."""
        runners = [load_text_detector(name) for name in detectors]
        logger.info(f"Loaded OCR detectors: {', '.join([repr(r) for r in runners])}")
        return cls(runners, device=device)

    def _scale_detections(
        self,
        detections: List[Detection],
        scale: Tuple[float, float],
    ) -> List[Detection]:
        for det in detections:
            det.bbox[0] *= scale[1]
            det.bbox[1] *= scale[0]
            det.bbox[2] *= scale[1]
            det.bbox[3] *= scale[0]
        return detections

    def process(
        self,
        image: np.ndarray,
        *,
        conf: float = 0.0,
        scale: Tuple[float, float] = (1.0, 1.0),
    ) -> List[Detection]:
        """Run OCR on an image and return scaled detections."""
        all_detections: List[Detection] = []

        for det in self.detectors:
            detections = det.run(image, conf=conf)

            all_detections.extend(detections)

        scaled_detections = self._scale_detections(all_detections, scale)
        return scaled_detections
