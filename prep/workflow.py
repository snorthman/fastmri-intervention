import logging, os, json
from pathlib import Path

import click

from prep.convert import dcm2mha, generate_dcm2mha_json, mha2nnunet, generate_mha2nnunet_json
from prep.upload import upload_data
from prep.annotate import write_annotations
from prep.dockerfile import Dockerfile
from prep.utils import now, remake_dir, DirectoryManager, GCAPI


def create_timestamp(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / '.timestamp', 'w') as f:
        f.write(now())


def consume_timestamp(d: Path) -> bool:
    t = d / '.timestamp'
    if b := (d / '.timestamp').exists():
        os.remove(t)
    return b


def summary(dm: DirectoryManager, archive_dir: Path, kwargs: dict):
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
{json.dumps(kwargs, indent=4)}
"""


def step_archive2mha(dm: DirectoryManager, archive_dir: Path):
    settings = Path(dm.output / 'dcm2mha_settings.json')
    if not settings.exists():
        logging.info(f'No dcm2mha_settings.json found, generating...')
        generate_dcm2mha_json(archive_dir, dm.output)

    logging.info(f'Converting raw archive @ {archive_dir} to mha files @ {dm.mha}')
    remake_dir(dm.mha)
    dcm2mha(archive_dir, dm.mha, settings)


def step_upload(dm: DirectoryManager, gc: GCAPI):
    logging.info(f'Uploading mha files @ {dm.mha} to grand-challenge.org/reader-studies/{gc.slug}')
    upload_data(dm.mha, gc)


def step_annotations(dm: DirectoryManager, gc: GCAPI):
    remake_dir(dm.annotations)
    write_annotations(dm.mha, dm.annotations, gc)


def step_mha2nnunet(dm: DirectoryManager, name: str, id: int):
    settings = Path(dm.output / 'mha2nnunet_settings.json')
    if not settings.exists():
        logging.info(f'No mha2nnunet_settings.json found, generating...')
        generate_mha2nnunet_json(dm.mha, dm.output, dm.output)

    logging.info(f'Converting mha @ {dm.mha} to nnunet structure @ {dm.nnunet}...')
    remake_dir(dm.nnunet)
    mha2nnunet(name, id, dm.mha, dm.annotations, dm.nnunet)


def workflow(**kwargs):
    logging.basicConfig(filename=f'fastmri-intervention_{now()}.log',
                        encoding='utf-8',
                        level=logging.INFO)

    try:
        out_dir = Path(kwargs['out_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
        archive_dir = Path(kwargs['archive_dir'])
    except:
        logging.critical(e := 'Output directory is required.' if 'out_dir' not in kwargs else 'Archive directory is required.')
        raise KeyError(e)
    dm = DirectoryManager(Path('.'), out_dir)

    s = summary(dm, archive_dir, kwargs)
    logging.debug(s)
    click.confirm(s, abort=True)

    invalidate = kwargs.get('invalidate', [])

    def check_invalidate(name: str):
        if b := name in invalidate:
            logging.info(f'{name} invalidated.')
        return b

    gc = GCAPI(kwargs.get('gc_slug', None), kwargs.get('gc_api', None))

    task_name = kwargs.get('task_name', 'fastmri_intervention')
    task_id = kwargs.get('task_id', 500)

    if not consume_timestamp(dm.mha) or check_invalidate('mha'):
        if not archive_dir:
            logging.critical('No valid MHA directory found, and no archive directory provided!')
            raise FileNotFoundError()
        step_archive2mha(dm, archive_dir)
    else:
        logging.info('Valid MHA directory found.')
    create_timestamp(dm.mha)

    if not consume_timestamp(dm.upload) or check_invalidate('upload'):
        if not archive_dir:
            logging.critical('No valid MHA directory found, and no archive directory provided!')
            raise FileNotFoundError()
        step_upload(dm, gc)
    else:
        logging.info('Upload skipped.')
    create_timestamp(dm.upload)

    if not consume_timestamp(dm.annotations) or check_invalidate('annotations'):
        step_annotations(dm, gc)
    else:
        logging.info('Valid annotation directory found.')
    create_timestamp(dm.annotations)

    if not consume_timestamp(dm.nnunet) or check_invalidate('nnunet'):
        step_mha2nnunet(dm, task_name, task_id)
    else:
        logging.info('Valid nnunet directory found.')
    create_timestamp(dm.nnunet)

    logging.info('Data preprocessing complete, building image...')
    dockerfile = Dockerfile(dm, task_name, task_id, version=kwargs.get('docker_version', 1))
    with open(dm.output / 'docker_build_and_push.sh', 'w') as sh:
        sh.write(dockerfile.commands())
    logging.info(f'Build and push image using\n{dockerfile.commands()}')