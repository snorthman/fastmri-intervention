from pathlib import Path

from my.workflow import workflow


def test_workflow():
    input = Path('//umcsanfsclp01.umcn.nl/radng_diag_prostate/archives/Prostate-mpMRI-ScientificArchive/RUMC')
    output = Path('tests/output')
    j = output / 'dcm2mha_settings.json'

    output.mkdir(exist_ok=True)

    workflow(dcm2mha_json=j, archive_dir=input, output_dir=output)