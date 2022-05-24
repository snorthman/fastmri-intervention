import shutil
from pathlib import Path
import pytest

import prep


def remake_dir(dir: Path) -> Path:
    if dir.exists():
        shutil.rmtree(dir)
    dir.mkdir()
    return dir


@pytest.fixture(scope="module")
def workflow():
    input = Path('//umcsanfsclp01.umcn.nl/radng_diag_prostate/archives/Prostate-mpMRI-ScientificArchive/RUMC/11670')
    output = Path('tests/output')
    j = Path('tests/input/dcm2mha_settings.json')
    slug = 'needle-segmentation-for-interventional-radiology'

    if not j.exists():
        prep.generate_dcm2mha_json(input, j.parent)

    try:
        with open(j.parent / 'api.txt') as f:
            api_key = f.readline()
    except FileNotFoundError:
        api_key = None

    return input, output, j, slug, api_key


def test_dcm2mha(workflow):
    input, output, j, _, _ = workflow
    output.mkdir(exist_ok=True)

    prep.dcm2mha(input, remake_dir(output / 'mha'), j)


def test_upload():
    pass


def test_annotations(workflow):
    input, output, _, slug, api_key = workflow
    output.mkdir(exist_ok=True)

    prep.write_annotations(output / 'mha', remake_dir(output / 'annotations'), slug, api_key)