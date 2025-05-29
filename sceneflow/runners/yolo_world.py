from typing import List, Sequence

import numpy as np
from ultralytics import YOLOWorld

from sceneflow.utils.hub import download_model_weights_to_zoo

from ._factory import OVD_DETECTORS
from ._helpers import Detection, ModelRunner


class YoloWorldRunner(ModelRunner):
    def __init__(self, model_name: str, device: str = "cpu"):
        self._classes = None
        super().__init__(model_name=model_name, device=device)

    def _load_model(self):
        path = download_model_weights_to_zoo(self.model_name) or self.model_name
        if not str(path).endswith(".pt"):
            path = str(path) + ".pt"
        self._model = YOLOWorld(path).to(self.device)

    def run(self, image: np.ndarray, texts: Sequence[str], conf: float = 0.5, **kwargs) -> List[Detection]:
        if self._classes is None:
            self._classes = texts
            self.model.set_classes(texts)

        results = self.model.predict(source=image, conf=conf, device=self.device, verbose=False)
        if not results or not results[0].boxes:
            return []

        boxes = results[0].boxes

        return [
            Detection(box, float(score), int(cls_idx), texts[int(cls_idx)])
            for box, score, cls_idx in zip(boxes.xyxy.cpu(), boxes.conf.cpu(), boxes.cls.int().cpu())
        ]


@OVD_DETECTORS.register("yolov8x-worldv2")
def yolov8x_worldv2(name: str = "yolov8x-worldv2", device: str = "cpu") -> YoloWorldRunner:
    return YoloWorldRunner(name, device=device)
