import random
from typing import List

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from sceneflow.runners._helpers import Detection


def random_color(seed=None):
    if seed is not None:
        random.seed(seed)
    return tuple(random.choices(range(50, 256), k=3))


def generate_static_scene_mask(image, detections):
    height, width = image.shape[:2]
    fg_mask = np.zeros((height, width), dtype=np.uint8)

    for det in detections:
        rle = det["segmentation"]
        if rle is not None:
            mask = mask_utils.decode(rle).astype(np.uint8)
            fg_mask = cv2.bitwise_or(fg_mask, mask)

    # Invert
    static_scene_mask = cv2.bitwise_not(fg_mask * 255)
    return static_scene_mask


def blend_detections(image: np.ndarray, detections: List[Detection], alpha: float = 1.0) -> np.ndarray:
    overlay = image.copy()

    for i, det in enumerate(detections):
        x0, y0, x1, y1 = det.xyxy
        label = det["class_name"]
        score = det["score"]
        mask = det["segmentation"]

        # Get color
        color = random_color(seed=i)

        if mask is not None:
            mask = mask_utils.decode(mask)

            if mask.shape != image.shape[:2]:
                raise ValueError(f"Mask shape {mask.shape} does not match image shape {image.shape[:2]}")

            mask_indices = mask.astype(bool)
            for c in range(3):
                overlay[:, :, c][mask_indices] = (
                    overlay[:, :, c][mask_indices] * (1 - alpha) + color[c] * alpha
                ).astype(np.uint8)

        # Draw bboxes
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color=color, thickness=2)

        # Add class label and confidence
        text = f"{label} {score:.2f}" if label else f"{score:.2f}"
        cv2.putText(
            overlay,
            text,
            (x0, max(y0 - 5, 10)),
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            0.30,
            (128, 128, 128),  # Dark gray for text
            1,
            cv2.LINE_AA,
        )

    # Blend overlay
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
