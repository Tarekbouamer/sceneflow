from pathlib import Path

import click

from sceneflow.pipelines.ocr import detect_text_boxes
from sceneflow.runners._factory import TEXT_DETECTORS


@click.command(name="ocr-detect")
@click.option("--input-dir", required=True, type=click.Path(exists=True), help="Directory with input images")
@click.option("--output-dir", required=True, type=click.Path(), help="Directory to save outputs")
@click.option(
    "--text-detector",
    type=click.Choice(TEXT_DETECTORS.list_models()),
    default="mmocr_dbnet_abinet",
    show_default=True,
    help="Text detector to use for OCR",
)
@click.option("--det-thd", default=0.5, type=float, show_default=True, help="Detection confidence threshold")
@click.option("--resize", type=(int, int), default=None, help="Resize images to (width, height)")
def ocr_cli(input_dir, output_dir, text_detector, det_thd, resize):
    """Run OCR-based text detection on a folder of images."""
    detect_text_boxes(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        text_detector=text_detector,
        det_thd=det_thd,
        resize=resize,
    )
