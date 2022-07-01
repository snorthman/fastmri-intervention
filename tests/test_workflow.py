import shutil, os, json
from pathlib import Path

import pytest

import prep.convert, prep.annotate, prep.workflow, prep.upload
from prep.utils import DirectoryManager, GCAPI


def remake_dir(dir: Path) -> Path:
    if dir.exists():
        shutil.rmtree(dir)
    dir.mkdir()
    return dir


def assert_dir(dir: Path, contents):
    for d in os.listdir(dir):
        if not d.endswith('.log'):
            assert d in contents


@pytest.fixture(scope="module")
def inputs():
    dm = DirectoryManager(Path('tests'), Path('output'))
    archive_dir = Path('tests/input/10880')
    slug = 'needle-segmentation-for-interventional-radiology'
    dm.output.mkdir(parents=True, exist_ok=True)

    try:
        with open('tests/input/api.txt') as f:
            api_key = f.readline()
    except FileNotFoundError:
        api_key = None
    gc = GCAPI(slug, api_key)

    return dm, archive_dir, gc


def test_dcm2mha(inputs):
    dm, archive_dir, _ = inputs

    prep.convert.dcm2mha(archive_dir, remake_dir(dm.mha))

    # specific to 10880
    assert_dir(dm.mha / '10880', ['10880_182386710290888504267667945338785981449_needle_0.mha',
                                  '10880_182386710290888504267667945338785981449_needle_1.mha',
                                  '10880_182386710290888504267667945338785981449_needle_2.mha',
                                  '10880_182386710290888504267667945338785981449_needle_3.mha',
                                  '10880_182386710290888504267667945338785981449_needle_4.mha',
                                  '10880_182386710290888504267667945338785981449_needle_5.mha',
                                  '10880_230637160173546023230130340285289177320_needle_0.mha',
                                  '10880_230637160173546023230130340285289177320_needle_1.mha',
                                  '10880_230637160173546023230130340285289177320_needle_2.mha',
                                  '10880_230637160173546023230130340285289177320_needle_3.mha',
                                  '10880_230637160173546023230130340285289177320_needle_4.mha',
                                  '10880_230637160173546023230130340285289177320_needle_5.mha',
                                  '10880_230637160173546023230130340285289177320_needle_6.mha',
                                  '10880_230637160173546023230130340285289177320_needle_7.mha',
                                  '10880_244375702689236279917785509476093985322_needle_0.mha',
                                  '10880_244375702689236279917785509476093985322_needle_1.mha',
                                  '10880_244375702689236279917785509476093985322_needle_2.mha',
                                  '10880_244375702689236279917785509476093985322_needle_3.mha'])


def test_upload(inputs):
    dm, archive_dir, gc = inputs
    # prep.upload.upload_data(dm.mha, gc)
    # prep.upload.delete_all_data(gc)


def test_annotations(inputs):
    dm, _, gc = inputs
    prep.annotate.write_annotations(dm.mha, remake_dir(dm.annotations), gc)

    # specific to 10880
    assert_dir(dm.annotations, ['10880_244375702689236279917785509476093985322_needle_1.nii.gz',
                                '10880_182386710290888504267667945338785981449_needle_5.nii.gz'])


def test_mha2nnunet(inputs):
    dm, _, _ = inputs

    prep.convert.mha2nnunet('fastmri_intervention', 500, dm.mha, dm.annotations, remake_dir(dm.nnunet))

    # specific to 10880
    # assert_dir(dm.nnunet, ['mha2nnunet_settings.json', 'Task500_fastmri_intervention'])
    # assert_dir(dm.nnunet, ['mha2nnunet_settings.json', 'Task500_fastmri_intervention'])
    # assert_dir(dm.nnunet / 'Task500_fastmri_intervention', ['dataset.json', 'imagesTr', 'labelsTr'])
    # assert_dir(dm.nnunet / 'Task500_fastmri_intervention/imagesTr', ['10880_182386710290888504267667945338785981449_0000.nii.gz',
    #                                                                                '10880_230637160173546023230130340285289177320_0000.nii.gz',
    #                                                                                '10880_244375702689236279917785509476093985322_0000.nii.gz'])
    # assert_dir(dm.nnunet / 'Task500_fastmri_intervention/labelsTr', ['10880_182386710290888504267667945338785981449.nii.gz',
    #                                                                                '10880_230637160173546023230130340285289177320.nii.gz',
    #                                                                                '10880_244375702689236279917785509476093985322.nii.gz'])


def test_prepare():
    with open('tests/input/workflow.json') as j:
        workflow = json.load(j)
    prep.workflow.workflow(**workflow)