from typing import List

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results as UltralyticsResults

from sceneflow.utils.hub import download_model_weights_to_zoo

from ._factory import DETECTORS
from ._helpers import Detection, ModelRunner


class YoloRunner(ModelRunner):
    def _load_model(self):
        path = download_model_weights_to_zoo(self.model_name) or self.model_name
        self._model = YOLO(str(path) + ".pt").to(self.device)

    def run(self, image: np.ndarray, conf: float = 0.25, **kwargs) -> List[Detection]:
        results: List[UltralyticsResults] = self.model.predict(
            source=image, conf=conf, device=self.device, verbose=False
        )
        if not results or not results[0].boxes:
            return []

        boxes = results[0].boxes
        names = getattr(self.model.model, "names", {})

        return [
            Detection(
                bbox=box,
                score=float(score),
                class_id=int(cls_id),
                class_name=names.get(int(cls_id), str(int(cls_id))),
            )
            for box, cls_id, score in zip(boxes.xyxy.cpu(), boxes.cls.int().cpu(), boxes.conf.cpu())
        ]


@DETECTORS.register("yolov8n")
def yolov8n():
    return YoloRunner("yolov8n")


@DETECTORS.register("yolov8x")
def yolov8x():
    return YoloRunner("yolov8x")
