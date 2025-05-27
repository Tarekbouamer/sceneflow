from __future__ import annotations

from functools import partial
from typing import Dict, List, Union

import cv2
import numpy as np

AVAILABLE_CAMOUFLAGE_METHODS = {
    "telea",
    "ns",
    "median",
    "blur",
    "mosaic",
    "solid",
    "noise",
}


def combine_masks(masks: List[np.ndarray]) -> np.ndarray:
    out_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for m in masks:
        out_mask = cv2.bitwise_or(out_mask, m)
    return out_mask


class Camouflage:
    """Fast privacy/redaction helper."""

    default_cfg = {
        "telea": {"radius": 3},
        "ns": {"radius": 3},
        "median": {"kernel": 21},
        "blur": {"ksize": 21},
        "mosaic": {"block": 16},
        "solid": {"value": (127, 127, 127)},
        "noise": {},
    }

    def __init__(self, method: str = "telea", cfg: Dict | None = None):
        self.method = method.lower()
        self.cfg = {**self.default_cfg, **(cfg or {})}
        self._runner = self._setup()

    def _setup(self):
        if self.method == "telea":
            return partial(cv2.inpaint, inpaintRadius=self.cfg["telea"]["radius"], flags=cv2.INPAINT_TELEA)

        if self.method == "ns":
            return partial(cv2.inpaint, inpaintRadius=self.cfg["ns"]["radius"], flags=cv2.INPAINT_NS)

        if self.method == "median":
            k = self.cfg["median"]["kernel"]

            def _median(img, m):
                blur = cv2.medianBlur(img, k)
                out = img.copy()
                out[m > 0] = blur[m > 0]
                return out

            return _median

        if self.method == "blur":
            k = self.cfg["blur"]["ksize"]

            def _gaussian(img, m):
                blur = cv2.GaussianBlur(img, (k, k), 0)
                out = img.copy()
                out[m > 0] = blur[m > 0]
                return out

            return _gaussian

        if self.method == "mosaic":
            block = self.cfg["mosaic"]["block"]

            def _mosaic(img, m):
                h, w = img.shape[:2]
                small = cv2.resize(img, (w // block, h // block), interpolation=cv2.INTER_LINEAR)
                big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                out = img.copy()
                out[m > 0] = big[m > 0]
                return out

            return _mosaic

        if self.method == "solid":
            value = self.cfg["solid"]["value"]

            def _solid(img, m):
                out = img.copy()
                out[m > 0] = value
                return out

            return _solid

        if self.method == "noise":

            def _noise(img, m):
                out = img.copy()
                noise = np.random.randint(0, 256, img.shape, dtype=np.uint8)
                out[m > 0] = noise[m > 0]
                return out

            return _noise

        raise ValueError(f"Unknown method '{self.method}'. Allowed: {list(self._defaults)}")

    @staticmethod
    def _norm_mask(mask: np.ndarray) -> np.ndarray:
        """Normalise mask to 0-255 uint8."""
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.max() == 1:
            mask = mask * 255
        return mask.astype(np.uint8)

    def hide(
        self,
        image: np.ndarray,
        masks: Union[np.ndarray, List[np.ndarray]],
    ) -> np.ndarray:
        """
        Hide pixels defined by *masks*.

        Args:
            image (np.ndarray): Input image.
            masks (Union[np.ndarray, List[np.ndarray]]): masks to hide HxW or HxWxC or list of HxW.
        """
        # normalise to list
        if isinstance(masks, list):
            mask_list = masks
        elif isinstance(masks, np.ndarray) and masks.ndim == 3:
            mask_list = [masks[i] for i in range(masks.shape[0])]
        else:
            mask_list = [masks]

        if len(mask_list) == 0:
            raise ValueError("No masks provided. Returning original image.")

        mask_list = [self._norm_mask(m) for m in mask_list]

        merged = combine_masks(mask_list)
        return self._runner(image, merged)
