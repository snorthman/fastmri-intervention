from pathlib import Path

import click

import intervention.prep.prep
import intervention.segment.inference
from intervention.utils import Settings


@click.group()
def cli():
    pass


@cli.command(name='prep')
@click.option('-s', '--settings', type=click.Path(resolve_path=True, path_type=Path),
              prompt='Enter path/to/settings.json', default='.', help="Path to json settings file")
def prep(settings: Path):
    intervention.prep.prep(Settings('prep', settings))


@cli.command(name='inference')
@click.option('-s', '--settings', type=click.Path(resolve_path=True, path_type=Path),
              prompt='Enter path/to/settings.json', default='.', help="Path to json settings file")
def inference(settings: Path):
    intervention.segment.inference(Settings('inference', settings))


@cli.command(name='plot')
@click.option('-s', '--settings', type=click.Path(resolve_path=True, path_type=Path),
              prompt='Enter path/to/settings.json', default='.', help="Path to json settings file")
def plot(settings: Path):
    s = Settings('plot', settings)
    intervention.segment.plot(s.dm.predict)