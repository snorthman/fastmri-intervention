import logging, json
from pathlib import Path
from datetime import datetime

import click

from intervention.prep.convert import dcm2mha, generate_dcm2mha_json, mha2nnunet
from intervention.prep.upload import upload_data, delete_all_data
from intervention.prep.annotate import write_annotations
from intervention.utils import DirectoryManager, GCAPI, Settings


def step_dcm(dm: DirectoryManager, archive_dir: Path):
    logging.info("\nSTEP_DCM\n")

    generate_dcm2mha_json(dm, archive_dir)


def step_dcm2mha(dm: DirectoryManager, archive_dir: Path):
    logging.info("\nSTEP_DCM2MHA\n")

    logging.info(f'Converting raw archive @ {archive_dir} to mha files @ {dm.mha} using {dm.dcm}/')
    dcm2mha(dm, archive_dir)


def step_upload(dm: DirectoryManager, gc: GCAPI):
    logging.info("\nSTEP_UPLOAD\n")

    logging.info(f'Deleting mha files @ grand-challenge.org/reader-studies/{gc.slug}')
    if click.confirm('Confirm delete? (required when uploading)'):
        delete_all_data(gc)
        logging.info(f'Uploading mha files @ {dm.mha} to grand-challenge.org/reader-studies/{gc.slug}')
        upload_data(dm.mha, gc)
    else:
        logging.info('Cancelled delete, skipping upload step')


def step_annotations(dm: DirectoryManager, gc: GCAPI):
    logging.info("\nSTEP_ANNOTATIONS\n")

    write_annotations(dm, gc)


def step_mha2nnunet(dm: DirectoryManager):
    logging.info("\nSTEP_MHA2NNUNET\n")

    logging.info(f'Converting mha @ {dm.mha} to nnunet structure @ {dm.nnunet}...')
    mha2nnunet(dm)


def prep(settings: Settings):
    start = datetime.now()
    logging.info(f"Program started at {start}")

    dm, gc, archive_dir = settings.dm, settings.gc, settings.archive_dir

    steps = settings.run_prep
    funcs = []
    for id, step in [('dcm', lambda: step_dcm(dm, archive_dir)),
                     ('dcm2mha', lambda: step_dcm2mha(dm, archive_dir)),
                     ('upload', lambda: step_upload(dm, gc)),
                     ('annotate', lambda: step_annotations(dm, gc)),
                     ('mha2nnunet', lambda: step_mha2nnunet(dm))]:
        if id in steps:
            funcs.append(step)

    s = settings.summary()
    logging.debug(s)
    click.confirm(s, abort=True)

    for func in funcs:
        func()

    end = datetime.now()
    logging.info(f"Program end at {end}\n\truntime {end - start}")