import click

from .redact import redact_cli


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """SceneFlow CLI."""
    pass


# Add sub-commands below
cli.add_command(redact_cli, name="redact")
