import shutil
from pathlib import Path

import pytest

import intervention.segment
from intervention.utils import Settings


def test_diagnose():
    predict_dir = Path('tests/output/predict')
    predict_dir.mkdir(exist_ok=True, parents=True)

    shutil.rmtree(predict_dir)
    shutil.copytree('tests/input/predict', predict_dir)

    settings = Settings('segment', Path('tests/input/settings.json'))
    setattr(settings.dm, 'output', Path('tests/input'))
    intervention.segment.predict(settings)
