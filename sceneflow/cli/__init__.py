import click

from .ocr_cli import ocr_cli
from .redact_cli import redact_cli


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """SceneFlow CLI."""
    pass


# Add sub-commands below
cli.add_command(redact_cli, name="redact")
cli.add_command(ocr_cli, name="ocr-detect")
