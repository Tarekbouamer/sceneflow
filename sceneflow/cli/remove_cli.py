from pathlib import Path

import click

from sceneflow.pipelines.remove import remove_objects_with_prompts
from sceneflow.runners._factory import INPAINTERS, OVD_DETECTORS, SEGMENTORS


@click.command(name="remove")
@click.option("--input-dir", required=True, type=click.Path(exists=True), help="Input image folder.")
@click.option("--output-dir", required=True, type=click.Path(), help="Folder to save inpainted outputs.")
@click.option(
    "--ovd-detector",
    type=click.Choice(OVD_DETECTORS.list_models()),
    default="owlvit_base",
    show_default=True,
    help="Open-vocabulary detector model to use.",
)
@click.option(
    "--segmentor",
    type=click.Choice(SEGMENTORS.list_models()),
    default="sam_h",
    show_default=True,
    help="Segmentation model to apply after detection.",
)
@click.option(
    "--inpainter",
    type=click.Choice(INPAINTERS.list_models()),
    default="big_lama",
    show_default=True,
    help="Inpainter model to use for object removal.",
)
@click.option(
    "--prompt", required=True, help="Comma-separated list of object class names to remove, e.g., 'person,car'."
)
@click.option("--det-thd", default=0.25, type=float, show_default=True, help="Detection confidence threshold.")
@click.option("--nms-iou", default=0.0, type=float, show_default=True, help="NMS IoU threshold.")
@click.option("--resize", type=(int, int), default=None, help="Resize images to (width height).")
def remove_cli(**kwargs):
    """Run OVD detection + SAM seg + LaMa inpainting to remove objects from images using prompts."""
    kwargs = {k: Path(v) if isinstance(v, click.Path) else v for k, v in kwargs.items()}

    remove_objects_with_prompts(**kwargs)
