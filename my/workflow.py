import os, click
from pathlib import Path

import fastmri_intervention


def prompt_dir(text: str, default: str = None) -> Path:
    return Path(click.prompt(text, type=click.Path), default=default).absolute()


if __name__ == '__main__':
    wd = Path(__file__).parent

    output = prompt_dir('Provide output directory:')
    mha_dir = output / 'mha'
    annotation_dir = output / 'annotations'
    nnunet_dir = output / 'nnunet'
    raw_archive = prompt_dir('Provide archive directory:', "//umcsanfsclp01.umcn.nl/radng_diag_prostate/archives/Prostate-mpMRI-ScientificArchive/RUMC")
    slug = click.prompt('Provide grandchallenge slug:', type=str, default='needle-segmentation-for-interventional-radiology')

    try:
        with open(wd / 'api.txt') as f:
            api_key = f.readline()
    except FileNotFoundError:
        api_key = None
    api = click.prompt('Provide grandchallenge api:', type=str, default=api_key)

    if click.confirm(f'Step 1: Convert raw archive to mha files @ {str(mha_dir)}'):
        fastmri_intervention.dcm2mha(raw_archive, mha_dir)
    if click.confirm(f'Step 2: Upload mha files to grand challenge @ grand-challenge.org/reader-studies/{slug}/'):
        click.echo('This action has already been performed and will be skipped!')
        # if False:
        #   fastmri_intervention.upload_data(mha_dir, slug, api)
    if click.confirm(f'Step 3: Download point annotations from grand challenge, and process into 3D annotations'):
