from typing import List

import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from sceneflow.runners._factory import TEXT_DETECTORS
from sceneflow.runners._helpers import Detection, ModelRunner


class TrOCRRunner(ModelRunner):
    """Performs end-to-end OCR using Microsoft's TrOCR model."""

    def _load_model(self):
        """Load the TrOCR model and processor."""
        self.processor = TrOCRProcessor.from_pretrained(self.model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device)

    def run(self, image: np.ndarray, conf: float = 0.0, **kwargs) -> List[Detection]:
        """Runs TrOCR on an image and returns the recognized text as a single detection."""
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return [
            Detection(
                bbox=np.array([0, 0, image.shape[1], image.shape[0]], dtype=np.float32),
                score=1.0,
                class_id=0,
                class_name="text",
                text=generated_text,
            )
        ]


@TEXT_DETECTORS.register("trocr_handwritten")
def trocr_handwritten():
    return TrOCRRunner("microsoft/trocr-base-handwritten")
