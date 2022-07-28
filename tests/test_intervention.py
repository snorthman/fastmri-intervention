import shutil, os
from pathlib import Path

import pytest

import intervention.dcm as dcm
import intervention.dcm2mha as dcm2mha
import intervention.mha2nnunet as mha2nnunet
import intervention.annotate as annotate
import intervention.inference as inference
from intervention.utils import CommandDCM, CommandPlot, CommandUpload, CommandAnnotate, CommandInference, \
    CommandMHA2nnUNet, CommandDCM2MHA, Settings


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
    if s.base.exists():
        shutil.rmtree(s.base)
    s = Settings(Path('input/settings.json'))

    assert len(s.commands) > 0
    # assert all([c.dm.output == Path('output') for c in s.commands])

    # s.commands[0].dm.output.mkdir(parents=True, exist_ok=True)

    return s


def test_dcm(inputs):
    s: Settings = inputs
    cmd: CommandDCM = find_command(s, 'dcm')

    dcm.generate_dcm2mha_json(cmd)


def test_dcm2mha(inputs):
    s: Settings = inputs
    test_dcm(inputs)
    cmd: CommandDCM2MHA = find_command(s, 'dcm2mha')

    dcm2mha.dcm2mha(cmd)

    # specific to 10880
    assert assert_dir(cmd.out_dir / '10880', '10880_182386710290888504267667945338785981449_needle_0.mha',
               '10880_182386710290888504267667945338785981449_needle_1.mha',
               '10880_182386710290888504267667945338785981449_needle_2.mha',
               '10880_182386710290888504267667945338785981449_needle_3.mha',
               '10880_244375702689236279917785509476093985322_needle_0.mha',
               '10880_244375702689236279917785509476093985322_needle_1.mha',
               '10880_244375702689236279917785509476093985322_needle_2.mha',
               '10880_244375702689236279917785509476093985322_needle_3.mha')


def test_annotations(inputs):
    s: Settings = inputs
    cmd: CommandAnnotate = find_command(s, 'annotate')

    annotate.write_annotations(cmd)

    # specific to 10880
    assert assert_dir(cmd.out_dir,
                      '10880_244375702689236279917785509476093985322_needle_1.nii.gz',
                      '10880_182386710290888504267667945338785981449_needle_5.nii.gz')


def test_mha2nnunet(inputs):
    s: Settings = inputs
    cmd: CommandMHA2nnUNet = find_command(s, 'mha2nnunet')

    mha2nnunet.mha2nnunet(cmd)

    # specific to 10880
    assert assert_dir(cmd.out_dir, 'mha2nnunet_train_settings.json', 'mha2nnunet_test_settings.json', 'nnunet_split.json')
    assert assert_dir(cmd.out_dir, cmd.task_dirname)
    assert assert_dir(cmd.out_dir / cmd.task_dirname, 'dataset.json', 'imagesTr', 'labelsTr', 'imagesTs', 'labelsTs')
    for niigz in ['10880_182386710290888504267667945338785981449_5_0000.nii.gz', '10880_244375702689236279917785509476093985322_1_0000.nii.gz']:
        assert any([assert_dir(cmd.out_dir / cmd.task_dirname / t, niigz) for t in ['imagesTr', 'imagesTs']])
    for niigz in ['10880_182386710290888504267667945338785981449_5.nii.gz', '10880_244375702689236279917785509476093985322_1.nii.gz']:
        assert any([assert_dir(cmd.out_dir / cmd.task_dirname / t, niigz) for t in ['labelsTr', 'labelsTs']])

def test_inference(inputs):
    s: Settings = inputs
    cmd: CommandInference = find_command(s, 'inference')

    assert Path('output/annotations').exists(), "run test_prep/test_annotations first"

    # predict_dir = cmd.in_dir.predict
    # predict_dir.mkdir(exist_ok=True, parents=True)
    #
    # shutil.rmtree(predict_dir)
    # shutil.copytree('input/predict', predict_dir)
    #
    # inference.inference(cmd)