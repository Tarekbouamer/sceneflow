from pathlib import Path

import click

from sceneflow.pipelines.redaction import redact


@click.command(name="redact")
@click.option("--input-dir", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--detector", default="rtdetr_l", show_default=True)
@click.option("--segmentor", default="sam_l", show_default=True)
@click.option("--det-thd", default=0.25, type=float, show_default=True)
@click.option("--allowed-classes", default=None, help="Comma-separated class names/IDs to keep")
@click.option("--camouflage-method", type=click.Choice(["telea", "ns"]), default="ns", show_default=True)
def redact_cli(**kwargs):
    """Run detection → segmentation → camouflage on a folder of images."""
    kwargs = {k: Path(v) if isinstance(v, click.Path) else v for k, v in kwargs.items()}
    redact(**kwargs)
