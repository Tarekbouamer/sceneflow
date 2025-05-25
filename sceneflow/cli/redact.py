from pathlib import Path

import click

from sceneflow.core.camouflage import AVAILABLE_CAMOUFLAGE_METHODS
from sceneflow.pipelines.redaction import redact
from sceneflow.runners._factory import SEGMENTORS


@click.command(name="redact")
@click.option("--input-dir", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
@click.option(
    "--detectors",
    multiple=True,
    default=("yolov8n", "rtdetr_l"),
    show_default=True,
    help="Closed-vocabulary detector model names",
)
@click.option(
    "--ovd-detectors",
    multiple=True,
    default=("owlvit_base",),
    help="Open-vocabulary detector model names",
)
@click.option("--segmentor", type=click.Choice(SEGMENTORS.list_models()), default="sam_h", show_default=True)
@click.option("--nms-iou", default=0.5, type=float, show_default=True)
@click.option("--det-thd", default=0.4, type=float, show_default=True)
@click.option("--allowed-classes", default=None, help="Comma-separated class names/IDs to keep")
@click.option(
    "--camouflage-method", type=click.Choice(AVAILABLE_CAMOUFLAGE_METHODS), default="solid", show_default=True
)
@click.option("--resize", type=(int, int), default=None, help="Resize images to (width, height)")
def redact_cli(**kwargs):
    """Run detection → segmentation → camouflage on a folder of images."""
    kwargs = {k: Path(v) if isinstance(v, click.Path) else v for k, v in kwargs.items()}
    redact(**kwargs)
