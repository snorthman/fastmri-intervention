import json
from pathlib import Path

import click

from .prep import prep
from .segment.predict import predict
from .utils import Settings


@click.group()
def cli():
    pass


@cli.command(name='prep')
@click.option('-s', '--settings', type=click.Path(resolve_path=True, path_type=Path),
              prompt='Enter path/to/settings.json', default='.', help="Path to json settings file")
def prep(settings: Path):
    prep(Settings('prep', settings))


@cli.command(name='segment')
@click.option('-s', '--settings', type=click.Path(resolve_path=True, path_type=Path),
              prompt='Enter path/to/settings.json', default='.', help="Path to json settings file")
def segment(settings: Path):
    with open(settings) as j:
        settings = json.load(j)
    predict(Settings('segment', settings))