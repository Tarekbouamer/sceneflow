from pathlib import Path
from typing import List

import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

from sceneflow.utils.hub import download_model_weights_to_zoo

from ._factory import SEGMENTORS
from ._helpers import Detection, ModelRunner


class SAMRunner(ModelRunner):
    def _load_model(self):
        variant = self.model_name.replace("sam_", "")
        ckpt = download_model_weights_to_zoo(f"sam_{variant}") or f"sam_{variant}"
        ckpt = Path(ckpt)
        if not ckpt.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {ckpt}")

        model_key = f"vit_{variant}"
        if model_key not in sam_model_registry:
            raise ValueError(f"Unsupported SAM variant '{variant}'")

        sam = sam_model_registry[model_key](checkpoint=str(ckpt)).to(self.device)
        self._model = SamPredictor(sam)

    def run(self, image: np.ndarray, detections: List[Detection], **kwargs) -> List[np.ndarray]:
        if not detections:
            return []

        predictor: SamPredictor = self.model
        predictor.set_image(image)

        boxes = torch.tensor([det.bbox.tolist() for det in detections], dtype=torch.float32)
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image.shape[:2]).to(predictor.device)

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        return masks.squeeze(1).cpu().numpy().astype(np.uint8)


@SEGMENTORS.register("sam_b")
def sam_b(name: str = "sam_b", device: str = "cpu") -> SAMRunner:
    return SAMRunner(name, device=device)


@SEGMENTORS.register("sam_l")
def sam_l(name: str = "sam_l", device: str = "cpu") -> SAMRunner:
    return SAMRunner(name, device=device)


@SEGMENTORS.register("sam_h")
def sam_h(name: str = "sam_h", device: str = "cpu") -> SAMRunner:
    return SAMRunner(name, device=device)
