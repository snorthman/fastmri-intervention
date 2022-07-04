import shutil, os, json
from pathlib import Path

import pytest

import prep.convert, prep.annotate, prep.workflow, prep.upload, prep.docker
from prep.utils import DirectoryManager, GCAPI


def remake_dir(dir: Path) -> Path:
    if dir.exists():
        shutil.rmtree(dir)
    dir.mkdir()
    return dir


def assert_dir(dir: Path, *contents):
    A = os.listdir(dir)
    for b in contents:
        assert b in A


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

    remake_dir(dm.mha)
    j = prep.convert.generate_dcm2mha_json(dm, archive_dir)
    prep.convert.dcm2mha(dm, archive_dir, j=j)

    # specific to 10880
    assert_dir(dm.mha / '10880', '10880_182386710290888504267667945338785981449_needle_0.mha',
               '10880_182386710290888504267667945338785981449_needle_1.mha',
               '10880_182386710290888504267667945338785981449_needle_2.mha',
               '10880_182386710290888504267667945338785981449_needle_3.mha',
               '10880_244375702689236279917785509476093985322_needle_0.mha',
               '10880_244375702689236279917785509476093985322_needle_1.mha',
               '10880_244375702689236279917785509476093985322_needle_2.mha',
               '10880_244375702689236279917785509476093985322_needle_3.mha')


def test_upload(inputs):
    dm, archive_dir, gc = inputs
    # prep.upload.upload_data(dm.mha, gc)
    # prep.upload.delete_all_data(gc)


def test_annotations(inputs):
    dm, _, gc = inputs

    remake_dir(dm.annotations)
    prep.annotate.write_annotations(dm, gc)

    # specific to 10880
    assert_dir(dm.annotations, '10880_244375702689236279917785509476093985322_needle_1.nii.gz',
                               '10880_182386710290888504267667945338785981449_needle_5.nii.gz')


def test_mha2nnunet(inputs):
    dm, _, _ = inputs

    remake_dir(dm.nnunet)
    j = prep.convert.generate_mha2nnunet_json(dm)
    prep.convert.mha2nnunet(dm, 'fastmri_intervention', 500, j=j)

    # specific to 10880
    taskdirname = 'Task500_fastmri_intervention'
    assert_dir(dm.output, 'mha2nnunet_settings.json', taskdirname)
    assert_dir(dm.nnunet, taskdirname)
    assert_dir(dm.nnunet / taskdirname, 'dataset.json', 'imagesTr', 'labelsTr')
    assert_dir(dm.nnunet / f'{taskdirname}/imagesTr', '10880_182386710290888504267667945338785981449_5_0000.nii.gz',
                                                      '10880_244375702689236279917785509476093985322_1_0000.nii.gz')
    assert_dir(dm.nnunet / f'{taskdirname}/labelsTr', '10880_182386710290888504267667945338785981449_5.nii.gz',
                                                      '10880_244375702689236279917785509476093985322_1.nii.gz')


def test_dockerfile(inputs):
    dm, _, _ = inputs
    df = prep.docker.Dockerfile(dm, 'fastmri_intervention', 500)



def test_prepare():
    with open('tests/input/workflow.json') as j:
        workflow = json.load(j)
    prep.workflow.workflow(**workflow)
