import pytest
from pathlib import Path

import fastmri_intervention.convert_data


@pytest.mark.order(0)
def test_prepare_data():
    input = Path('//umcsanfsclp01.umcn.nl/radng_diag_prostate/archives/Prostate-mpMRI-ScientificArchive/RUMC/10007')
    output = Path('tests/output')
    j = Path('tests/output/dcm2mha_settings.json')

    if not j.exists():
        fastmri_intervention.convert_data.generate_dcm2mha_json(input, output)

    fastmri_intervention.convert_data.dcm2mha(input, output, j if j.exists() else None)

@pytest.mark.order(1)
def test_upload_data():
    pass