import httpx
from pathlib import Path

import gcapi, click

import fastmri_intervention


def prompt_dir(text: str, default: str = None) -> Path:
    return Path(click.prompt(text, type=click.Path), default=default).absolute()


def workflow(dcm2mha_json: Path = None, output_dir: Path = None, archive_dir: Path = None, slug: str = None, auto: bool = False):
    wd = Path(__file__).parent

    output_dir = prompt_dir('Provide output directory:') if not output_dir else output_dir
    mha_dir = output_dir / 'mha'
    annotation_dir = output_dir / 'annotations'
    nnunet_dir = output_dir / 'nnunet'

    raw_archive_default = '//umcsanfsclp01.umcn.nl/radng_diag_prostate/archives/Prostate-mpMRI-ScientificArchive/RUMC'
    raw_archive = prompt_dir('Provide archive directory:', raw_archive_default) if not archive_dir else archive_dir

    slug_default = 'needle-segmentation-for-interventional-radiology'
    slug = click.prompt('Provide grandchallenge slug:', type=str, default=slug_default) if not slug else slug

    try:
        with open(wd / 'api.txt') as f:
            api_key = f.readline()
    except FileNotFoundError:
        api_key = None

    api_key = click.prompt('Provide grandchallenge api:', type=str, default=api_key) if not api_key else api_key
    try:
        next(gcapi.Client(token=api_key).reader_studies.iterate_all(params={"slug": slug}))
    except httpx.HTTPStatusError as e:
        raise ConnectionRefusedError(f'Invalid api key!\n\n{e}')

    if auto or click.confirm(f'Step 1: Convert raw archive to mha files @ {str(mha_dir)}\n\n'):
        fastmri_intervention.dcm2mha(raw_archive, mha_dir, dcm2mha_json)
    if auto or click.confirm(f'Step 2: Upload mha files to grand challenge @ grand-challenge.org/reader-studies/{slug}/\n\n'):
        click.echo('Upload has already been performed and will be skipped!')
        # if False:
        #   fastmri-intervention.upload_data(mha_dir, slug, api)
    if auto or click.confirm(f'Step 3: Download point annotations from grand challenge, and process into 3D annotations\n\n'):
        fastmri_intervention.write_annotations(mha_dir, annotation_dir, slug, api_key)
