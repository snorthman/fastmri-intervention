import pytest, json
from pathlib import Path

import fastmri_intervention.prepare_data


@pytest.mark.order(0)
def test_prepare_data():
    input = Path('//umcsanfsclp01.umcn.nl/radng_diag_prostate/archives/Prostate-mpMRI-ScientificArchive/RUMC/10007')
    output = Path('tests/output')
    j = Path('tests/output/dcm2mha_settings.json')

    if not j.exists():
        fastmri_intervention.prepare_data.generate_dcm2mha_json(input, output)

    # with open(j) as f:
    #     jsonfile = json.load(f)
    # with open(j, 'w') as f:
    #     jsonfile['archive'] = jsonfile['archive'][:100]
    #     json.dump(jsonfile, f, indent=4)

    fastmri_intervention.prepare_data.dcm2mha(input, output, j if j.exists() else None)

@pytest.mark.order(1)
def test_upload_data():
    pass