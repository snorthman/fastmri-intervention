import logging, json
from pathlib import Path
from datetime import datetime

import click, jsonschema

from prep.convert import dcm2mha, generate_dcm2mha_json, mha2nnunet, generate_mha2nnunet_jsons
from prep.upload import upload_data, delete_all_data
from prep.annotate import write_annotations
from prep.utils import now, DirectoryManager, GCAPI, workflow_schema


def create_timestamp(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    with open(path / '.timestamp', 'w') as f:
        f.write(now())


def summary(dm: DirectoryManager, archive_dir: Path, jsons: str):
    def dir_exists(d: Path, error = False) -> str:
        if d.exists() and d.is_dir():
            return '<<EXISTS>>'
        if error:
            logging.critical(e := f'{d} does not exist or is not a directory.')
            raise NotADirectoryError(e)
        else:
            return ''
    summary_settings = lambda s: f'{s}: {dm.output / f"{s}_settings.json"} {"<<EXISTS>>" if (dm.output / f"{s}_settings.json").exists() else "<<MISSING>>"}'

    summary_archive_dir = f'{archive_dir.absolute()}  {dir_exists(archive_dir)}' if archive_dir else 'NO ARCHIVE DIR'
    summary_dm = '\n'.join([f'{n}: {d.absolute()}  {dir_exists(d)}' for n, d in [('mha dir', dm.mha), ('annotations dir', dm.annotations), ('nnunet dir', dm.nnunet)]])
    return f"""DIRECTORIES:
output dir: {dm.output.absolute()}  {dir_exists(dm.output, error=True)}
archive dir: {summary_archive_dir}

{summary_dm}

SETTINGS:
{summary_settings('dcm2mha')}
{summary_settings('mha2nnunet')}

JSON:
{jsons}
"""


def step_dcm2mha(dm: DirectoryManager, archive_dir: Path):
    logging.info("STEP_DCM2MHA")
    create_timestamp(dm.mha)

    settings = Path(dm.output / 'dcm2mha_settings.json')
    if not settings.exists():
        logging.info(f'No dcm2mha_settings.json found, generating...')
        generate_dcm2mha_json(dm, archive_dir)

    logging.info(f'Converting raw archive @ {archive_dir} to mha files @ {dm.mha}')
    dcm2mha(dm, archive_dir, settings)


def step_upload(dm: DirectoryManager, gc: GCAPI):
    logging.info("STEP_UPLOAD")
    create_timestamp(dm.upload)

    logging.info(f'Deleting mha files @ grand-challenge.org/reader-studies/{gc.slug}')
    if click.confirm('Confirm delete? (required when uploading)'):
        delete_all_data(gc)
        logging.info(f'Uploading mha files @ {dm.mha} to grand-challenge.org/reader-studies/{gc.slug}')
        upload_data(dm.mha, gc)
    else:
        logging.info('Cancelled delete, skipping upload step')


def step_annotations(dm: DirectoryManager, gc: GCAPI):
    logging.info("STEP_ANNOTATIONS")
    create_timestamp(dm.annotations)

    write_annotations(dm, gc)


def step_mha2nnunet(dm: DirectoryManager, name: str, id: int):
    logging.info("STEP_MHA2NNUNET")
    create_timestamp(dm.nnunet)

    train = Path(dm.output / 'mha2nnunet_train_settings.json')
    test = Path(dm.output / 'mha2nnunet_test_settings.json')
    if not train.exists() or not test.exists():
        logging.info(f'No mha2nnunet_settings.jsons found, generating...')
        train, test = generate_mha2nnunet_jsons(dm, name, id)

    logging.info(f'Converting mha @ {dm.mha} to nnunet structure @ {dm.nnunet}...')
    mha2nnunet(dm, train, test)


def workflow(**kwargs):
    logging.basicConfig(filename=f'fastmri-intervention_{now()}.log',
                        encoding='utf-8',
                        level=logging.INFO)
    start = datetime.now()
    logging.info(f"Program start at {start}")

    jsonschema.validate(kwargs, workflow_schema, jsonschema.Draft7Validator)

    archive_dir = Path(kwargs['archive_dir'])
    dm = DirectoryManager(Path('.'), Path(kwargs['out_dir']))

    gc = GCAPI(kwargs['gc_slug'], kwargs['gc_api'])

    steps = kwargs['run']
    funcs = []
    for id, step in [('dcm2mha', lambda: step_dcm2mha(dm, archive_dir)),
                 ('upload', lambda: step_upload(dm, gc)),
                 ('annotate', lambda: step_annotations(dm, gc)),
                 ('mha2nnunet', lambda: step_mha2nnunet(dm, kwargs['task_name'], kwargs['task_id']))]:
        if id in steps or 'all' in steps:
            funcs.append(step)

    s = summary(dm, archive_dir, json.dumps(kwargs, indent=4))
    logging.debug(s)
    click.confirm(s, abort=True)

    for func in funcs:
        func()

    end = datetime.now()
    logging.info(f"Program end at {end}\n\truntime {end - start}")