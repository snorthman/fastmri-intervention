import shutil
from pathlib import Path

import pytest

import intervention.segment
from intervention.utils import Settings


def test_inference():
    assert Path('tests/output/annotations').exists(), "run test_prep/test_annotations first"

    predict_dir = Path('tests/output/predict')
    predict_dir.mkdir(exist_ok=True, parents=True)

    shutil.rmtree(predict_dir)
    shutil.copytree('tests/input/predict', predict_dir)

    settings = Settings(Path('tests/input/settings.json'))
    setattr(settings.dm, 'output', Path('tests/input'))
    intervention.segment.inference(settings)


# def test_plot():
#     inference_dir = Path('tests/output/predict_results')
#     inference_dir.mkdir(exist_ok=True, parents=True)
#
#     shutil.rmtree(inference_dir)
#     shutil.copytree('tests/input/predict_results', inference_dir)
#
#     intervention.segment.plot(dm)
#
#     assert (inference_dir / f'{inference_dir.name}.png').exists()