from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from loguru import logger
from pycocotools import mask as mask_utils
from torchvision.ops import nms

from sceneflow.runners._factory import (
    load_detector,
    load_ovd_detector,
    load_segmentor,
)
from sceneflow.runners._helpers import Detection


class MaskGenerator:
    def __init__(
        self,
        detectors: List,
        ovd_detectors: List,
        segmentor,
        *,
        device: str,
    ) -> None:
        self.detectors = detectors
        self.ovd_detectors = ovd_detectors
        self.segmentor = segmentor
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        detectors: Sequence[str],
        ovd_detectors: Sequence[str],
        segmentor: str,
        *,
        device: str = "cuda:0",
    ) -> "MaskGenerator":
        """
        Instantiate a MaskGenerator with specific detector and segmentor names.
        Models are loaded through the centralized registry system.
        """
        #
        detector_runners = [load_detector(name) for name in detectors]
        logger.info(f"Loaded detectors: {', '.join([repr(r) for r in detector_runners])}")

        ovd_runners = [load_ovd_detector(name) for name in ovd_detectors]
        logger.info(f"Loaded OVD detectors: {', '.join([repr(r) for r in ovd_runners])}")

        segmentor_runner = load_segmentor(segmentor)
        logger.info(f"Loaded segmentor: {repr(segmentor_runner)}")

        return cls(detector_runners, ovd_runners, segmentor_runner, device=device)

    def _scale(self, detections: List[Detection], masks: np.ndarray, scale: Tuple[float, float]):
        for det in detections:
            det.bbox[0] *= scale[1]
            det.bbox[1] *= scale[0]
            det.bbox[2] *= scale[1]
            det.bbox[3] *= scale[0]

        resized_masks = np.array(
            [
                cv2.resize(
                    mask,
                    (int(mask.shape[1] * scale[1]), int(mask.shape[0] * scale[0])),
                    interpolation=cv2.INTER_NEAREST,
                )
                for mask in masks
            ]
        )
        return detections, resized_masks

    def _nms(self, detections: List[Detection], nms_iou: float = 0.5) -> List[Detection]:
        if not detections:
            return []

        boxes = torch.stack([d.bbox_tensor for d in detections])
        scores = torch.tensor([d.score for d in detections])
        ids = torch.tensor([d.class_id for d in detections])

        keep = nms(
            boxes,
            scores,
            iou_threshold=nms_iou,
        )
        return [detections[i] for i in keep]

    def _detect(
        self,
        image: np.ndarray,
        *,
        allowed_classes: Sequence[str],
        conf: float,
        nms_iou: float = 0.5,
    ) -> List[Detection]:
        out: List[Detection] = []

        for det in self.detectors:
            out.extend(det.run(image, conf=conf))

        # Filter detections by allowed classes
        if allowed_classes:
            out = [d for d in out if d.class_name in allowed_classes]

        for ovd_det in self.ovd_detectors:
            out.extend(ovd_det.run(image, texts=allowed_classes, conf=conf))

        return self._nms(out, nms_iou=nms_iou)

    def _segment(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        if not detections:
            return np.array([])
        return self.segmentor.run(image, detections=detections)

    def _to_rle(self, detections: List[Detection], masks: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        if len(detections) == 0 or len(masks) == 0:
            return detections, masks

        for det, mask in zip(detections, masks):
            det.mask = mask
            rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")
            det.segmentation = rle

        return detections, masks

    def generate(
        self,
        image: np.ndarray,
        *,
        conf: float = 0.25,
        nms_iou: float = 0.5,
        allowed_classes: Sequence[str] = None,
        scale: Tuple[float, float] = (1.0, 1.0),
    ) -> Tuple[List[Dict], np.ndarray, List[str]]:
        # Allow classes
        allowed_classes = list(set(allowed_classes)) if allowed_classes else None

        # Detect objects
        detections = self._detect(image, allowed_classes=allowed_classes, conf=conf, nms_iou=nms_iou)

        if not detections:
            return [], np.array([]), []

        masks = self._segment(image, detections)
        detections, masks = self._scale(detections, masks, scale)
        detections, masks = self._to_rle(detections, masks)

        prompts = sorted({d.class_name for d in detections})

        assert len(detections) == len(masks), "Number of detections and masks must match."
        return detections, masks, prompts
