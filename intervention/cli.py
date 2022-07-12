import json
from pathlib import Path

import click

import intervention.prep.prep
import intervention.segment.predict
from intervention.utils import Settings


@click.group()
def cli():
    pass


@cli.command(name='prep')
@click.option('-s', '--settings', type=click.Path(resolve_path=True, path_type=Path),
              prompt='Enter path/to/settings.json', default='.', help="Path to json settings file")
def prep(settings: Path):
    intervention.prep.prep(Settings('prep', settings))


@cli.command(name='segment')
@click.option('-s', '--settings', type=click.Path(resolve_path=True, path_type=Path),
              prompt='Enter path/to/settings.json', default='.', help="Path to json settings file")
def segment(settings: Path):
    with open(settings) as j:
        settings = json.load(j)
    intervention.segment.predict(Settings('segment', settings))