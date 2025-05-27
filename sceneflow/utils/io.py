import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .logger import logger


def get_all_images(input_dir: Path, exts=(".jpg", ".jpeg", ".png")):
    """Recursively collect all image files from a directory."""
    input_dir = Path(input_dir)
    images = [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]
    images = sorted(images)
    logger.info(f"Found {len(images)} image(s) in {input_dir}")
    return images


def load_image(path: Path, resize: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load an image using OpenCV and return it."""

    # Read image
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img_bgr is None:
        logger.error(f"Unable to read image: {path}")
        raise FileNotFoundError(path)

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    original_size = img.shape[:2]

    if resize is None:
        return img, img_bgr, original_size, np.array([1.0, 1.0], dtype=np.float32)

    # Resize image
    w_tgt, h_tgt = resize
    img = cv2.resize(img, (w_tgt, h_tgt), interpolation=cv2.INTER_LINEAR)
    scale = np.array([original_size[0] / h_tgt, original_size[1] / w_tgt], dtype=np.float32)

    return img, img_bgr, original_size, scale


def save_image(image: np.ndarray, save_path: Path):
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(str(save_path), image)


def save_mask(mask: np.ndarray, save_path: Path):
    if isinstance(mask, list):
        mask = np.array(mask)

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # To 255
    if mask.max() == 1 and mask.ndim == 2:
        mask *= 255

    cv2.imwrite(str(save_path), mask)


def save_annotations(data: dict, save_path: Path):
    json_path = save_path.with_suffix(".json")

    try:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved JSON: {json_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON: {json_path} — {e}")
