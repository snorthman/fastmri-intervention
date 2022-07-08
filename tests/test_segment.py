import shutil, os, json
from pathlib import Path

import pytest

import intervention.segment


def test_diagnose():
    with open('tests/input/settings.json') as j:
        settings = json.load(j)
    intervention.segment.diagnose(**settings)
