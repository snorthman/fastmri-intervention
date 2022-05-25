import shutil, os, json
from pathlib import Path

import pytest

import prep.convert, prep.annotate, prep.workflow


def remake_dir(dir: Path) -> Path:
    if dir.exists():
        shutil.rmtree(dir)
    dir.mkdir()
    return dir


def assert_dir(dir: Path, contents):
    assert [d for d in os.listdir(dir) if not d.endswith('.log')] == contents


def test_prepare():
    with open('tests/input/workflow.json') as j:
        workflow = json.load(j)
    prep.workflow.workflow(pelvis=Path('.'), radng_diag_prostate=Path('//umcsanfsclp01.umcn.nl/radng_diag_prostate'), **workflow)


@pytest.fixture(scope="module")
def inputs():
    input = Path('//umcsanfsclp01.umcn.nl/radng_diag_prostate/archives/Prostate-mpMRI-ScientificArchive/RUMC/10880')
    output = Path('tests/output')
    input_files = Path('tests/input')
    slug = 'needle-segmentation-for-interventional-radiology'

    try:
        with open(input_files / 'api.txt') as f:
            api_key = f.readline()
    except FileNotFoundError:
        api_key = None

    return input, output, input_files, slug, api_key


def test_dcm2mha(inputs):
    input, output, input_json, _, _ = inputs
    dcm2mha_json = input_json / 'dcm2mha_settings.json'
    output.mkdir(exist_ok=True)

    prep.convert.dcm2mha(input, remake_dir(output / 'mha'), dcm2mha_json)

    # specific to 10880
    assert_dir(output / 'mha/10880', ['10880_182386710290888504267667945338785981449_trufi.mha',
                                                '10880_230637160173546023230130340285289177320_trufi.mha',
                                                '10880_244375702689236279917785509476093985322_trufi.mha'])


def test_upload():
    pass


def test_annotations(inputs):
    input, output, _, slug, api_key = inputs
    output.mkdir(exist_ok=True)

    prep.annotate.write_annotations(output / 'mha', remake_dir(output / 'annotations'), slug, api_key)

    # specific to 10880
    assert_dir(output / 'annotations', ['10880_182386710290888504267667945338785981449_trufi.nii.gz',
                                                  '10880_230637160173546023230130340285289177320_trufi.nii.gz',
                                                  '10880_244375702689236279917785509476093985322_trufi.nii.gz'])


def test_mha2nnunet(inputs):
    input, output, input_json, _, _ = inputs
    nnunet2mha_json = input_json / 'mha2nnunet_settings.json'
    output.mkdir(exist_ok=True)

    prep.convert.mha2nnunet(output / 'mha', output / 'annotations', remake_dir(output / 'nnunet'), nnunet2mha_json)

    # specific to 10880
    assert_dir(output / 'nnunet', ['mha2nnunet_settings.json', 'Task500_fastmri_intervention'])
    assert_dir(output / 'nnunet', ['mha2nnunet_settings.json', 'Task500_fastmri_intervention'])
    assert_dir(output / 'nnunet/Task500_fastmri_intervention', ['dataset.json', 'imagesTr', 'labelsTr'])
    assert_dir(output / 'nnunet/Task500_fastmri_intervention/imagesTr', ['10880_182386710290888504267667945338785981449_0000.nii.gz',
                                                                                   '10880_230637160173546023230130340285289177320_0000.nii.gz',
                                                                                   '10880_244375702689236279917785509476093985322_0000.nii.gz'])
    assert_dir(output / 'nnunet/Task500_fastmri_intervention/labelsTr', ['10880_182386710290888504267667945338785981449.nii.gz',
                                                                                   '10880_230637160173546023230130340285289177320.nii.gz',
                                                                                   '10880_244375702689236279917785509476093985322.nii.gz'])
