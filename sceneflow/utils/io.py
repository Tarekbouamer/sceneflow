import json
from pathlib import Path

import cv2
import numpy as np

from .logger import logger


def get_all_images(input_dir: Path, exts=(".jpg", ".jpeg", ".png")):
    """Recursively collect all image files from a directory."""
    input_dir = Path(input_dir)
    images = [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]
    logger.info(f"Found {len(images)} image(s) in {input_dir}")
    return images


def load_image(path: Path):
    """Load an image using OpenCV and return it."""
    img = cv2.imread(str(path))
    if img is None:
        logger.warning(f"Failed to load image: {path}")
    return img


def save_image(image: np.ndarray, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    success = cv2.imwrite(str(save_path), image)
    if success:
        logger.info(f"Saved image: {save_path}")
    else:
        logger.error(f"Failed to save image: {save_path}")


def save_mask(mask: np.ndarray, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(mask, list):
        mask = np.array(mask)

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # To 255
    if mask.max() == 1 and mask.ndim == 2:
        mask *= 255

    success = cv2.imwrite(str(save_path), mask)
    if success:
        logger.info(f"Saved mask: {save_path}")
    else:
        logger.error(f"Failed to save mask: {save_path}")


def save_annotations(data: dict, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = save_path.with_suffix(".json")

    try:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved JSON: {json_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON: {json_path} — {e}")
