import httpx, logging, os
from pathlib import Path

import gcapi

from prep.convert import dcm2mha, generate_dcm2mha_json, mha2nnunet, generate_mha2nnunet_json
from prep.upload import upload_data
from prep.annotate import write_annotations
from prep.dockerfile import Dockerfile
from prep.utils import now, remake_dir, DirectoryManager


def create_timestamp(d: Path):
    with open(d / '.timestamp', 'w') as f:
        f.write(now())


def consume_timestamp(d: Path) -> bool:
    t = d / '.timestamp'
    if b := (d / '.timestamp').exists():
        os.remove(t)
    return b


def step_archive2mha(dm: DirectoryManager, archive_dir: Path):
    settings = Path(dm.output / 'dcm2mha_settings.json')
    if not settings.exists():
        logging.info(f'No dcm2mha_settings.json found, generating...')
        generate_dcm2mha_json(archive_dir, dm.output)

    logging.info(f'Converting raw archive @ {archive_dir} to mha files @ {dm.mha}')
    remake_dir(dm.mha)
    dcm2mha(archive_dir, dm.mha, settings)


def step_annotations(dm: DirectoryManager, gc_slug: str, gc_api: str):
    logging.info(f'Testing connection to grand-challenge.org ...')
    try:
        next(gcapi.Client(token=gc_api).reader_studies.iterate_all(params={"slug": gc_slug}))
    except httpx.HTTPStatusError as e:
        raise ConnectionRefusedError(f'Invalid api key!\n\n{e}')
    logging.info('Connected.')

    remake_dir(dm.annotations)
    write_annotations(dm.mha, dm.annotations, gc_slug, gc_api)


def step_mha2nnunet(dm: DirectoryManager, name: str, id: int):
    settings = Path(dm.output / 'mha2nnunet_settings.json')
    if not settings.exists():
        logging.info(f'No mha2nnunet_settings.json found, generating...')
        generate_mha2nnunet_json(dm.mha, dm.output, dm.output)

    logging.info(f'Converting mha @ {dm.mha} to nnunet structure @ {dm.nnunet}...')
    remake_dir(dm.nnunet)
    mha2nnunet(name, id, dm.mha, dm.annotations, dm.nnunet)


def workflow(pelvis: Path, radng_diag_prostate: Path = None, **kwargs) -> Dockerfile:
    logging.basicConfig(filename=f'fastmri-intervention_{now()}.log',
                        encoding='utf-8',
                        level=logging.DEBUG if kwargs.get('debug', False) else logging.INFO)

    if not (out_dir := kwargs.get('out_dir', None)):
        logging.critical(e := 'Output directory is required.')
        raise KeyError(e)
    dm = DirectoryManager(pelvis, out_dir)

    invalidate = kwargs.get('invalidate', [])

    def check_invalidate(name: str):
        if b := name in invalidate:
            logging.info(f'{name} invalidated.')
        return b

    archive_dir = kwargs.get('archive_dir', None)
    if archive_dir:
        archive_dir = radng_diag_prostate / archive_dir

    gc_slug = kwargs.get('gc_slug', None)
    gc_api = kwargs.get('gc_api', None)

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

    if not consume_timestamp(dm.annotations) or check_invalidate('annotation'):
        if not gc_slug or gc_api:
            logging.critical('No valid annotations directory found, and no slug or api key provided!')
            raise FileNotFoundError()
        step_annotations(dm, gc_slug, gc_api)
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