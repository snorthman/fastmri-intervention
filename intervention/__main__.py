import json
from pathlib import Path

import click

import intervention.prep, intervention.segment


@click.group()
def cli():
    pass


@cli.command(name='prep')
@click.option('-s', '--settings', type=click.Path(resolve_path=True, path_type=Path),
              prompt='Enter path/to/settings.json', default='.', help="Path to json settings file")
def prep(settings: Path):
    with open(settings) as j:
        settings = json.load(j)
    intervention.prep.prep(**settings)


@cli.command(name='diagnose')
@click.option('-s', '--settings', type=click.Path(resolve_path=True, path_type=Path),
              prompt='Enter path/to/settings.json', default='.', help="Path to json settings file")
def prep(settings: Path):
    with open(settings) as j:
        settings = json.load(j)
    intervention.segment.diagnose(**settings)