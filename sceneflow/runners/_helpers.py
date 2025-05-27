import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch


class ModelRunner:
    """Generic base class for model runners."""

    def __init__(self, model_name: Optional[str] = None, device: str = "cuda:0"):
        self.model_name = model_name or self.__class__.__name__
        self.device = device
        self._model = None
        self._load_model()

    def _load_model(self):
        """Load the model instance. Must be implemented by subclasses."""
        raise NotImplementedError("Inherited classes must implement `_load_model()`")

    def run(self, *args, **kwargs):
        """Execute the model's core function (e.g., predict, extract)."""
        raise NotImplementedError("Inherited classes must implement `run()`")

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def __repr__(self):
        return f"({self.__class__.__name__} model={self.model_name} device={self.device})"

    @property
    def model(self) -> Any:
        """Returns the loaded model instance, loading if necessary."""
        if self._model is None:
            self._load_model()
        return self._model


@dataclass
class Detection:
    """
    Unified detection result structure.
    """

    bbox: np.ndarray
    score: float
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    segmentation: Dict[str, Any] = None
    text: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.bbox.tolist(),
            "score": self.score,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "segmentation": self.segmentation,
            "text": self.text,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict())

    @property
    def xyxy(self) -> np.ndarray:
        return np.array(self.bbox, dtype=np.int32)

    @property
    def xywh(self) -> np.ndarray:
        x0, y0, x1, y1 = self.bbox
        return np.array([x0, y0, x1 - x0, y1 - y0], dtype=np.float32)

    @property
    def bbox_int(self) -> np.ndarray:
        return np.array(self.bbox, dtype=np.int32)

    @property
    def bbox_tensor(self) -> torch.Tensor:
        bbox_tensor = torch.tensor(self.bbox, dtype=torch.float32)
        return bbox_tensor.clone().detach()

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Detection":
        """Create a Detection instance from a dictionary."""
        return cls(
            bbox=np.array(d["bbox"], dtype=np.float32),
            score=float(d["score"]),
            class_id=int(d["class_id"]),
            class_name=str(d["class_name"]),
            segmentation=d.get("segmentation"),
            text=d.get("text"),
        )
