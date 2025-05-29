import os
from typing import List, Sequence

import numpy as np
import torch
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from ._factory import OVD_DETECTORS
from ._helpers import Detection, ModelRunner

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise EnvironmentError("Environment variable 'HF_TOKEN' is not set. Please set it to your Hugging Face token.")


class OwlViTRunner(ModelRunner):
    def _load_model(self):
        self._processor = OwlViTProcessor.from_pretrained(self.model_name, token=HF_TOKEN)
        self._model = OwlViTForObjectDetection.from_pretrained(self.model_name, token=HF_TOKEN).to(self.device)

    def run(self, image: np.ndarray, texts: Sequence[str], conf: float = 0.25, **kwargs) -> List[Detection]:
        processor = self._processor
        model = self.model

        inputs = processor(text=texts, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.shape[:2]], device=self.device)
        results = processor.post_process_object_detection(outputs, threshold=conf, target_sizes=target_sizes)[0]

        return [
            Detection(
                bbox=box,
                score=float(score),
                class_id=int(label),
                class_name=texts[int(label)],
            )
            for box, score, label in zip(results["boxes"].cpu(), results["scores"].cpu(), results["labels"].cpu())
        ]


@OVD_DETECTORS.register("owlvit_base")
def owlvit_base(name: str = "google/owlvit-base-patch32", device: str = "cpu") -> OwlViTRunner:
    return OwlViTRunner(name, device=device)


@OVD_DETECTORS.register("owlvit_large")
def owlvit_large(name: str = "google/owlvit-large-patch14", device: str = "cpu") -> OwlViTRunner:
    return OwlViTRunner(name, device=device)
