from typing import List, Optional

import numpy as np
from mmocr.apis import MMOCRInferencer
from mmocr.utils.polygon_utils import poly2bbox

from sceneflow.runners._factory import TEXT_DETECTORS
from sceneflow.runners._helpers import Detection, ModelRunner
from sceneflow.utils.stdout_utils import suppress_stdout_stderr


class MMOCRRunner(ModelRunner):
    """Generic MMOCR runner that accepts detector and recognizer configuration."""

    def __init__(self, name: str, det: str = "DBNet", rec: Optional[str] = "CRNN", **kwargs):
        self.det = det
        self.rec = rec
        super().__init__(name, **kwargs)

    def _load_model(self):
        self._model = MMOCRInferencer(
            det=self.det,
            rec=self.rec,
            device=self.device,
        )

    @suppress_stdout_stderr()
    def run(self, image: np.ndarray, conf: float = 0.5, **kwargs) -> List[Detection]:
        results = self._model(image, show=False)

        detections: List[Detection] = []

        for item in results["predictions"]:
            polygons = item.get("det_polygons", [])
            det_scores = item.get("det_scores", [])
            rec_scores = item.get("rec_scores", [])
            texts = item.get("rec_texts", [])

            if not (len(polygons) == len(det_scores) == len(texts) == len(rec_scores)):
                continue

            for polygon, det_score, text, rec_score in zip(polygons, det_scores, texts, rec_scores):
                if float(det_score) < conf or float(rec_score) < conf:
                    continue

                bbox = poly2bbox(polygon)

                detections.append(
                    Detection(
                        bbox=np.array(bbox, dtype=np.float32),
                        score=float(det_score),
                        class_id=0,
                        class_name="text",
                        text=text,
                    )
                )

        return detections


@TEXT_DETECTORS.register("mmocr_dbnet_crnn")
def mmocr_dbnet_crnn():
    return MMOCRRunner("mmocr_dbnet_crnn", det="DBNet", rec="CRNN")


@TEXT_DETECTORS.register("mmocr_dbnet_abinet")
def mmocr_dbnet_abinet():
    return MMOCRRunner("mmocr_dbnet_abinet", det="DBNet", rec="ABINet")


@TEXT_DETECTORS.register("mmocr_drrg_abinet")
def mmocr_drrg_abinet():
    return MMOCRRunner("mmocr_drrg_abinet", det="DRRG", rec="ABINet")


@TEXT_DETECTORS.register("mmocr_fcenet_crnn")
def mmocr_fcenet_crnn():
    return MMOCRRunner("mmocr_fcenet_crnn", det="FCENet", rec="CRNN")
