from typing import List

import numpy as np
from ultralytics import RTDETR
from ultralytics.engine.results import Results as UltralyticsResults

from sceneflow.utils.hub import download_model_weights_to_zoo

from ._factory import DETECTORS
from ._helpers import Detection, ModelRunner


class RTDETRRunner(ModelRunner):
    def _load_model(self):
        path = download_model_weights_to_zoo(self.model_name) or f"{self.model_name}.pt"
        self._model = RTDETR(str(path)).to(self.device)

    def run(self, image: np.ndarray, conf: float = 0.25, **kwargs) -> List[Detection]:
        results: List[UltralyticsResults] = self.model.predict(source=image, conf=conf, verbose=False)
        if not results or not results[0].boxes:
            return []

        boxes = results[0].boxes
        names = getattr(self.model.model, "names", {})

        return [
            Detection(
                bbox=box.cpu().numpy(),
                score=float(score),
                class_id=int(cls_id),
                class_name=names.get(int(cls_id), str(int(cls_id))),
            )
            for box, cls_id, score in zip(boxes.xyxy.cpu(), boxes.cls.int().cpu(), boxes.conf.cpu())
        ]


@DETECTORS.register("rtdetr_l")
def rtdetr_l(name: str = "rtdetr_l", device: str = "cpu") -> RTDETRRunner:
    return RTDETRRunner(name, device=device)


@DETECTORS.register("rtdetr_xl")
def rtdetr_xl(name: str = "rtdetr_l", device: str = "cpu") -> RTDETRRunner:
    return RTDETRRunner(name, device=device)
