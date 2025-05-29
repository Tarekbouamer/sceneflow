from __future__ import annotations

import numpy as np

from sceneflow.runners._factory import load_inpainter


class Remover:
    def __init__(self, inpainter: str = "big_lama", *, device: str = "cpu") -> None:
        """Initialize the remover with a specific inpainting model."""
        self.inpainter = load_inpainter(inpainter, device=device)

    def remove(self, image: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Remove objects from an image using the specified binary masks, each mask is applied sequentially."""

        if masks.ndim == 2:
            masks = [masks]
        elif masks.ndim == 3:
            masks = list(masks)
        else:
            raise ValueError(f"Expected masks of shape (H, W) or (N, H, W), got {masks.shape}")

        for mask in masks:
            assert mask.ndim == 2, f"Each mask must be a 2D array, got shape {mask.shape}"
            image = self.inpainter.run(image, mask)

        # Convert BGR 
        return image[..., ::-1]
