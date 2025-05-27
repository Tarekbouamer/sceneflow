from typing import List

import numpy as np
import pytesseract
from pytesseract import Output

from sceneflow.runners._factory import TEXT_DETECTORS

from ._helpers import Detection, ModelRunner


class TesseractRunner(ModelRunner):
    def _load_model(self):
        self.config = "--oem 3 --psm 6"

    def run(self, image: np.ndarray, conf: float = 0.0, **kwargs) -> List[Detection]:
        # Run Tesseract OCR on the input image
        data = pytesseract.image_to_data(image, config=self.config, output_type=Output.DICT)

        # Ensure confidence is in percentage
        conf = conf * 100 if conf < 1.0 else conf

        detections = []
        for i in range(len(data["text"])):
            text = data["text"][i]
            if not text.strip():
                continue
            score = float(data["conf"][i])
            if score < conf:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            bbox_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)

            detections.append(
                Detection(
                    bbox=bbox_xyxy,
                    score=score / 100.0,  # Convert to [0, 1] range
                    text=text,
                )
            )
        return detections


# Register useful PSM modes from 0 to 13 (excluding deprecated or rarely used ones)
PSM_MODES = {
    0: "osd_only",  # Orientation and script detection only
    1: "auto_with_osd",  # Automatic page segmentation with OSD
    3: "auto",  # Fully automatic page segmentation, no OSD
    4: "single_column",  # Assume a single column of text
    6: "single_block",  # Assume a single uniform block of text
    7: "single_line",  # Treat the image as a single text line
    8: "single_word",  # Treat the image as a single word
    11: "sparse",  # Sparse text with OSD
    13: "raw_line",  # Raw line — no layout analysis
}


@TEXT_DETECTORS.register("tesseract_osd_only")
def tesseract_osd_only():
    """Tesseract OCR with PSM 0 (OSD only)."""
    runner = TesseractRunner("tesseract_osd_only")
    runner.config = "--oem 3 --psm 0"
    return runner


@TEXT_DETECTORS.register("tesseract_auto_with_osd")
def tesseract_auto_with_osd():
    """Tesseract OCR with PSM 1 (Automatic page segmentation with OSD)."""
    runner = TesseractRunner("tesseract_auto_with_osd")
    runner.config = "--oem 3 --psm 1"
    return runner


@TEXT_DETECTORS.register("tesseract_auto")
def tesseract_auto():
    """Tesseract OCR with PSM 3 (Fully automatic page segmentation, no OSD)."""
    runner = TesseractRunner("tesseract_auto")
    runner.config = "--oem 3 --psm 3"
    return runner


@TEXT_DETECTORS.register("tesseract_single_column")
def tesseract_single_column():
    """Tesseract OCR with PSM 4 (Assume a single column of text)."""
    runner = TesseractRunner("tesseract_single_column")
    runner.config = "--oem 3 --psm 4"
    return runner


@TEXT_DETECTORS.register("tesseract_single_block")
def tesseract_single_block():
    """Tesseract OCR with PSM 6 (Assume a single uniform block of text)."""
    runner = TesseractRunner("tesseract_single_block")
    runner.config = "--oem 3 --psm 6"
    return runner


@TEXT_DETECTORS.register("tesseract_single_line")
def tesseract_single_line():
    """Tesseract OCR with PSM 7 (Treat the image as a single text line)."""
    runner = TesseractRunner("tesseract_single_line")
    runner.config = "--oem 3 --psm 7"
    return runner


@TEXT_DETECTORS.register("tesseract_single_word")
def tesseract_single_word():
    """Tesseract OCR with PSM 8 (Treat the image as a single word)."""
    runner = TesseractRunner("tesseract_single_word")
    runner.config = "--oem 3 --psm 8"
    return runner


@TEXT_DETECTORS.register("tesseract_sparse")
def tesseract_sparse():
    """Tesseract OCR with PSM 11 (Sparse text with OSD)."""
    runner = TesseractRunner("tesseract_sparse")
    runner.config = "--oem 3 --psm 11"
    return runner


@TEXT_DETECTORS.register("tesseract_raw_line")
def tesseract_raw_line():
    """Tesseract OCR with PSM 13 (Raw line — no layout analysis)."""
    runner = TesseractRunner("tesseract_raw_line")
    runner.config = "--oem 3 --psm 13"
    return runner


@TEXT_DETECTORS.register("tesseract_default")
def tesseract_default():
    """Tesseract OCR with default settings (PSM 3)."""
    runner = TesseractRunner("tesseract_default")
    runner.config = ""
    return runner
