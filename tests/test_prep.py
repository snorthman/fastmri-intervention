import shutil, os
from pathlib import Path

import pytest

import intervention.prep.convert as convert
import intervention.prep.annotate as annotate
from intervention.utils import DirectoryManager, GCAPI, Settings


def remake_dir(dir: Path) -> Path:
    if dir.exists():
        shutil.rmtree(dir)
    dir.mkdir()
    return dir


def assert_dir(dir: Path, *contents):
    A = os.listdir(dir)
    for b in contents:
        if b not in A:
            return False
    return True


@pytest.fixture(scope="module")
def inputs():
    dm = DirectoryManager(Path('tests'), Path('output'), 'fastmri_intervention', 500)
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


def test_dcm(inputs):
    dm, archive_dir, _ = inputs

    remake_dir(dm.dcm)

    convert.generate_dcm2mha_json(dm, archive_dir)


def test_dcm2mha(inputs):
    dm, archive_dir, _ = inputs

    remake_dir(dm.mha)

    convert.dcm2mha(dm, archive_dir)

    # specific to 10880
    assert assert_dir(dm.mha / '10880', '10880_182386710290888504267667945338785981449_needle_0.mha',
               '10880_182386710290888504267667945338785981449_needle_1.mha',
               '10880_182386710290888504267667945338785981449_needle_2.mha',
               '10880_182386710290888504267667945338785981449_needle_3.mha',
               '10880_244375702689236279917785509476093985322_needle_0.mha',
               '10880_244375702689236279917785509476093985322_needle_1.mha',
               '10880_244375702689236279917785509476093985322_needle_2.mha',
               '10880_244375702689236279917785509476093985322_needle_3.mha')


def test_annotations(inputs):
    dm, _, gc = inputs

    remake_dir(dm.annotations)
    annotate.write_annotations(dm, gc)

    # specific to 10880
    assert assert_dir(dm.annotations,
                      '10880_244375702689236279917785509476093985322_needle_1.nii.gz',
                      '10880_182386710290888504267667945338785981449_needle_5.nii.gz')


def test_mha2nnunet(inputs):
    dm, _, _ = inputs

    remake_dir(dm.nnunet)

    convert.mha2nnunet(dm)

    # specific to 10880
    assert assert_dir(dm.nnunet, 'mha2nnunet_train_settings.json', 'mha2nnunet_test_settings.json', 'nnunet_split.json')
    assert assert_dir(dm.nnunet, dm.task_dirname)
    assert assert_dir(dm.nnunet / dm.task_dirname, 'dataset.json', 'imagesTr', 'labelsTr', 'imagesTs', 'labelsTs')
    for niigz in ['10880_182386710290888504267667945338785981449_5_0000.nii.gz', '10880_244375702689236279917785509476093985322_1_0000.nii.gz']:
        assert any([assert_dir(dm.nnunet / dm.task_dirname / t, niigz) for t in ['imagesTr', 'imagesTs']])
    for niigz in ['10880_182386710290888504267667945338785981449_5.nii.gz', '10880_244375702689236279917785509476093985322_1.nii.gz']:
        assert any([assert_dir(dm.nnunet / dm.task_dirname / t, niigz) for t in ['labelsTr', 'labelsTs']])

