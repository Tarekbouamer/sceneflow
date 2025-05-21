from importlib.metadata import version

import click


@click.command()
def version_cmd():
    click.echo(version("sceneflow"))
