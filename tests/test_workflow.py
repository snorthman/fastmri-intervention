import shutil
from pathlib import Path

from my.workflow import workflow


def test_workflow():
    input = Path('//umcsanfsclp01.umcn.nl/radng_diag_prostate/archives/Prostate-mpMRI-ScientificArchive/RUMC')
    output = Path('tests/output')
    j = Path('tests/input/dcm2mha_settings.json')
    slug = 'needle-segmentation-for-interventional-radiology'

    shutil.rmtree(output)
    output.mkdir()

    # needs custom json
    workflow(dcm2mha_json=j, archive_dir=input, output_dir=output, slug=slug, auto=True)