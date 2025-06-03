from typing import Dict, List

import numpy as np
from PIL import Image


class ImageManager:
    """Manage one image and its annotation state (rects + resized preview)."""

    def __init__(self, filename: str):
        self._filename = filename
        self._img = Image.open(filename)
        self._rects: List[Dict] = []  # initial empty or externally provided
        self._current_rects: List[Dict] = []
        self._resized_ratio_w = 1.0
        self._resized_ratio_h = 1.0

    def set_rects(self, rects: List[Dict]) -> None:
        """Set raw image-space rects (from annotation store or new session)."""
        self._rects = rects

    def get_img(self) -> Image.Image:
        return self._img

    def get_rects(self) -> List[Dict]:
        return self._rects

    def resizing_img(self, max_height: int = 700, max_width: int = 700) -> Image.Image:
        resized = self._img.copy()

        if resized.height > max_height:
            ratio = max_height / resized.height
            resized = resized.resize(
                (int(resized.width * ratio), int(resized.height * ratio)),
                Image.Resampling.LANCZOS,
            )

        if resized.width > max_width:
            ratio = max_width / resized.width
            resized = resized.resize(
                (int(resized.width * ratio), int(resized.height * ratio)),
                Image.Resampling.LANCZOS,
            )

        self._resized_ratio_w = self._img.width / resized.width
        self._resized_ratio_h = self._img.height / resized.height
        return resized

    def _resize_rect(self, rect: Dict) -> Dict:
        resized = {
            "left": rect["left"] / self._resized_ratio_w,
            "width": rect["width"] / self._resized_ratio_w,
            "top": rect["top"] / self._resized_ratio_h,
            "height": rect["height"] / self._resized_ratio_h,
        }
        if "label" in rect:
            resized["label"] = rect["label"]
        return resized

    def get_resized_rects(self) -> List[Dict]:
        return [self._resize_rect(r) for r in self._rects]

    def _chop_box_img(self, rect: Dict):
        left = int(rect["left"] * self._resized_ratio_w)
        top = int(rect["top"] * self._resized_ratio_h)
        width = int(rect["width"] * self._resized_ratio_w)
        height = int(rect["height"] * self._resized_ratio_h)

        arr = np.asarray(self._img).astype("uint8")
        crop = arr[top : top + height, left : left + width]
        img = Image.fromarray(crop)

        return img, rect.get("label", "")

    def init_annotation(self, rects: List[Dict]) -> List:
        self._current_rects = rects[:]
        return [self._chop_box_img(r) for r in self._current_rects]

    def set_annotation(self, index: int, label: str) -> None:
        if 0 <= index < len(self._current_rects):
            self._current_rects[index]["label"] = label

    def get_current_rects(self) -> List[Dict]:
        return self._current_rects
