from pathlib import Path

import pytest

import intervention.segment
from intervention.utils import Settings


def test_diagnose():
    intervention.segment.diagnose(Settings('segment', Path('tests/input/settings.json')))
