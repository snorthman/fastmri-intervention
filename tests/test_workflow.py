import shutil, os, json
from pathlib import Path

import pytest

import prep.convert, prep.annotate, prep.workflow
from prep.utils import DirectoryManager


def remake_dir(dir: Path) -> Path:
    if dir.exists():
        shutil.rmtree(dir)
    dir.mkdir()
    return dir


def assert_dir(dir: Path, contents):
    assert [d for d in os.listdir(dir) if not d.endswith('.log')] == contents


@pytest.fixture(scope="module")
def inputs():
    dm = DirectoryManager('tests/output', '.')
    archive_dir = Path('//umcsanfsclp01.umcn.nl/radng_diag_prostate/archives/Prostate-mpMRI-ScientificArchive/RUMC/10880')
    slug = 'needle-segmentation-for-interventional-radiology'
    dm.output.mkdir(exist_ok=True)

    try:
        with open('tests/input/api.txt') as f:
            api_key = f.readline()
    except FileNotFoundError:
        api_key = None

    return dm, archive_dir, slug, api_key


def test_dcm2mha(inputs):
    dm, archive_dir, _, _ = inputs

    prep.convert.dcm2mha(archive_dir, remake_dir(dm.mha))

    # specific to 10880
    assert_dir(dm.mha / '10880', ['10880_182386710290888504267667945338785981449_trufi.mha',
                                                '10880_230637160173546023230130340285289177320_trufi.mha',
                                                '10880_244375702689236279917785509476093985322_trufi.mha'])


def test_upload():
    pass


def test_annotations(inputs):
    dm, _, slug, api_key = inputs
    prep.annotate.write_annotations(dm.mha, remake_dir(dm.annotations), slug, api_key)

    # specific to 10880
    assert_dir(dm.annotations, ['10880_182386710290888504267667945338785981449_trufi.nii.gz',
                                                  '10880_230637160173546023230130340285289177320_trufi.nii.gz',
                                                  '10880_244375702689236279917785509476093985322_trufi.nii.gz'])


def test_mha2nnunet(inputs):
    dm, _, _, _= inputs

    prep.convert.mha2nnunet('fastmri_intervention', 500, dm.mha, dm.annotations, remake_dir(dm.nnunet))

    # specific to 10880
    assert_dir(dm.nnunet, ['mha2nnunet_settings.json', 'Task500_fastmri_intervention'])
    assert_dir(dm.nnunet, ['mha2nnunet_settings.json', 'Task500_fastmri_intervention'])
    assert_dir(dm.nnunet / 'Task500_fastmri_intervention', ['dataset.json', 'imagesTr', 'labelsTr'])
    assert_dir(dm.nnunet / 'Task500_fastmri_intervention/imagesTr', ['10880_182386710290888504267667945338785981449_0000.nii.gz',
                                                                                   '10880_230637160173546023230130340285289177320_0000.nii.gz',
                                                                                   '10880_244375702689236279917785509476093985322_0000.nii.gz'])
    assert_dir(dm.nnunet / 'Task500_fastmri_intervention/labelsTr', ['10880_182386710290888504267667945338785981449.nii.gz',
                                                                                   '10880_230637160173546023230130340285289177320.nii.gz',
                                                                                   '10880_244375702689236279917785509476093985322.nii.gz'])


def test_prepare():
    with open('tests/input/workflow.json') as j:
        workflow = json.load(j)
    prep.workflow.workflow(pelvis=Path('.'), radng_diag_prostate=Path('//umcsanfsclp01.umcn.nl/radng_diag_prostate'), **workflow)