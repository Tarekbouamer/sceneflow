from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from pycocotools import mask as mask_utils
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from ultralytics import RTDETR, YOLO, YOLOWorld

from sceneflow.utils.hub import download_model_weights_to_zoo


def load_detector(model_name: str):
    """Return an Ultralytics detector."""
    name = Path(model_name).stem.lower()
    model_path = download_model_weights_to_zoo(name) or model_name

    if name.startswith("yolo"):
        return YOLO(model_path if str(model_path).endswith(".pt") else str(model_path) + ".pt")
    if name.startswith("rtdetr"):
        return RTDETR(model_path)
    if "yoloworld" in name or "world" in name:
        return YOLOWorld(model_path)

    raise ValueError(f"Unknown detector model '{model_name}'")


def load_segmentor(model_name: str, device: str = "cuda:0") -> Tuple[SamPredictor, SamAutomaticMaskGenerator]:
    """
    Load a SAM segmentor.
    """
    ckpt_path = download_model_weights_to_zoo(Path(model_name).stem.lower()) or Path(model_name)
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {ckpt_path}")

    variant = model_name.replace("sam_", "vit_")

    if variant not in sam_model_registry:
        raise ValueError(f"SAM variant '{variant}' not supported, available: {list(sam_model_registry.keys())}")

    sam = sam_model_registry[variant](checkpoint=str(ckpt_path)).to(device)
    return SamPredictor(sam), SamAutomaticMaskGenerator(sam)


class MaskGenerator:
    """Detector + SAM predictor wrapper."""

    def __init__(self, detector, segmentor: SamPredictor, device: str = "cuda:0"):
        self.detector = detector
        self.segmentor = segmentor
        self.device = device

    def _detect(
        self,
        image: np.ndarray,
        det_thd: float = 0.25,
        allowed_classes: Union[List[str], List[int], None] = None,
    ) -> Tuple[torch.Tensor, List[Dict], List[str]]:
        # Detector
        results = self.detector.predict(source=image, conf=det_thd, device=self.device, verbose=False)

        if not results or not results[0].boxes:
            return torch.empty((0, 4)), [], []

        boxes = results[0].boxes.xyxy.cpu()
        cls_ids = results[0].boxes.cls.int().cpu()
        confs = results[0].boxes.conf.cpu()
        names = getattr(self.detector.model, "names", {})

        kept = []
        for box, cid, conf in zip(boxes, cls_ids, confs):
            cname = names.get(int(cid), str(int(cid)))
            if allowed_classes and cid.item() not in allowed_classes and cname not in allowed_classes:
                continue
            kept.append((box, int(cid), cname, float(conf)))

        if not kept:
            return torch.empty((0, 4)), [], []

        bboxes = torch.stack([k[0] for k in kept])
        detections = [
            {
                "class_id": k[1],
                "class_name": k[2],
                "confidence": round(k[3], 4),
                "bbox": list(map(int, k[0].tolist())),
            }
            for k in kept
        ]
        prompts = sorted({k[2] for k in kept})
        return bboxes, detections, prompts

    @staticmethod
    def _embed_rle(detections: List[Dict], masks: np.ndarray) -> List[Dict]:
        for d, m in zip(detections, masks):
            rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")
            d["segmentation"] = rle
        return detections

    def _segment(self, image: np.ndarray, det_boxes: torch.Tensor) -> np.ndarray:
        """Segment the image using SAM."""
        if det_boxes.numel() == 0:
            return np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8)

        self.segmentor.set_image(image)
        tb = self.segmentor.transform.apply_boxes_torch(det_boxes, image.shape[:2]).to(self.device)

        # Generate masks
        masks, _, _ = self.segmentor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=tb,
            multimask_output=False,
        )
        return masks.squeeze(1).cpu().numpy().astype(np.uint8)

    def generate(
        self,
        image: np.ndarray,
        det_thd: float = 0.25,
        allowed_classes: Union[List[str], List[int], None] = None,
    ) -> Dict:
        # Detect objects
        boxes, detections, prompts = self._detect(image, det_thd, allowed_classes)

        if boxes.numel() == 0:
            h, w = image.shape[:2]
            return {
                "detections": [],
                "masks": np.zeros((0, h, w), dtype=np.uint8),
                "prompts": [],
            }

        # Segment
        masks = self._segment(image, boxes)

        # Update detection
        detections = self._embed_rle(detections, masks)

        return detections, masks, prompts
