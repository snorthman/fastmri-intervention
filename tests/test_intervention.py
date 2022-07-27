import shutil, os
from pathlib import Path

import pytest

import intervention.convert as convert
import intervention.annotate as annotate
import intervention.inference as inference
from intervention.utils import Command, Settings


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


def find_command(s: Settings, name: str):
    find = [c.name == name for c in s.commands]
    return s.commands[find.index(True)]


@pytest.fixture(scope="module")
def inputs():
    s = Settings(Path('input/settings.json'))

    assert len(s.commands) > 0
    assert all([c.dm.output == Path('output') for c in s.commands])

    s.commands[0].dm.output.mkdir(parents=True, exist_ok=True)

    return s


def test_dcm(inputs):
    s, = inputs
    c: Command = find_command(s, 'dcm')

    remake_dir(c.dm.dcm)

    convert.generate_dcm2mha_json(c.dm, c.archive_dir)


def test_dcm2mha(inputs):
    s, = inputs
    c: Command = find_command(s, 'dcm2mha')

    remake_dir(c.dm.mha)

    convert.dcm2mha(c.dm, c.archive_dir)

    # specific to 10880
    assert assert_dir(c.dm.mha / '10880', '10880_182386710290888504267667945338785981449_needle_0.mha',
               '10880_182386710290888504267667945338785981449_needle_1.mha',
               '10880_182386710290888504267667945338785981449_needle_2.mha',
               '10880_182386710290888504267667945338785981449_needle_3.mha',
               '10880_244375702689236279917785509476093985322_needle_0.mha',
               '10880_244375702689236279917785509476093985322_needle_1.mha',
               '10880_244375702689236279917785509476093985322_needle_2.mha',
               '10880_244375702689236279917785509476093985322_needle_3.mha')


def test_annotations(inputs):
    s, = inputs
    c: Command = find_command(s, 'annotate')

    remake_dir(c.dm.annotations)
    annotate.write_annotations(c.dm, c.gc)

    # specific to 10880
    assert assert_dir(c.dm.annotations,
                      '10880_244375702689236279917785509476093985322_needle_1.nii.gz',
                      '10880_182386710290888504267667945338785981449_needle_5.nii.gz')


def test_mha2nnunet(inputs):
    s, = inputs
    c: Command = find_command(s, 'mha2nnunet')

    remake_dir(c.dm.nnunet)

    convert.mha2nnunet(c.dm)

    # specific to 10880
    assert assert_dir(c.dm.nnunet, 'mha2nnunet_train_settings.json', 'mha2nnunet_test_settings.json', 'nnunet_split.json')
    assert assert_dir(c.dm.nnunet, c.dm.task_dirname)
    assert assert_dir(c.dm.nnunet / c.dm.task_dirname, 'dataset.json', 'imagesTr', 'labelsTr', 'imagesTs', 'labelsTs')
    for niigz in ['10880_182386710290888504267667945338785981449_5_0000.nii.gz', '10880_244375702689236279917785509476093985322_1_0000.nii.gz']:
        assert any([assert_dir(c.dm.nnunet / c.dm.task_dirname / t, niigz) for t in ['imagesTr', 'imagesTs']])
    for niigz in ['10880_182386710290888504267667945338785981449_5.nii.gz', '10880_244375702689236279917785509476093985322_1.nii.gz']:
        assert any([assert_dir(c.dm.nnunet / c.dm.task_dirname / t, niigz) for t in ['labelsTr', 'labelsTs']])

def test_inference():
    s, = inputs
    c: Command = find_command(s, 'inference')

    assert Path('tests/output/annotations').exists(), "run test_prep/test_annotations first"

    predict_dir = c.dm.predict
    predict_dir.mkdir(exist_ok=True, parents=True)

    shutil.rmtree(predict_dir)
    shutil.copytree('tests/input/predict', predict_dir)

    inference.inference(c)