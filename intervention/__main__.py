import logging
from pathlib import Path
from datetime import datetime

import click

from intervention.dcm import generate_dcm2mha_json
from intervention.dcm2mha import dcm2mha
from intervention.mha2nnunet import mha2nnunet
from intervention.upload import upload_data, delete_all_data
from intervention.annotate import write_annotations
from intervention.utils import DirectoryManager, GCAPI, Settings
# from intervention.inference import inference, plot


def upload(dm: DirectoryManager, gc: GCAPI):
    if click.confirm('Confirm delete? (required when uploading)'):
        logging.info(f'Deleting mha files @ grand-challenge.org/reader-studies/{gc.slug}')
        delete_all_data(gc)
        logging.info(f'Uploading mha files @ {dm.mha} to grand-challenge.org/reader-studies/{gc.slug}')
        upload_data(dm.mha, gc)
    else:
        logging.info('Cancelled delete, skipping upload step')


@click.command()
@click.option('-s', '--settings', type=click.Path(resolve_path=True, path_type=Path),
              prompt='Enter path/to/settings.json', default='.', help="Path to json settings file")
def cli(settings: Path):
    s = Settings(settings)
    print(s.summary())
    click.confirm('\nStart program?', abort=True)

    start = datetime.now()
    logging.info(f"Program started at {start}")

    for cmd in s.commands:
        if cmd.name == 'dcm':
            generate_dcm2mha_json(cmd)
        if cmd.name == 'dcm2mha':
            dcm2mha(cmd)
        if cmd.name == 'upload':
            pass
        if cmd.name == 'annotate':
            write_annotations(cmd)
        if cmd.name == 'mha2nnunet':
            mha2nnunet(cmd)
        # if cmd.name == 'inference':
        #     inference(cmd.dm, cmd.trainer)
        # if cmd.name == 'plot':
        #     plot(cmd.dm)

    end = datetime.now()
    logging.info(f"Program end at {end}\n\truntime {end - start}")


if __name__ == '__main__':
    cli()