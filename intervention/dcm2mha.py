from pathlib import Path

import picai_prep

from intervention.utils import DirectoryManager


def dcm2mha(dm: DirectoryManager, archive_dir: Path):
    picai_prep.Dicom2MHAConverter(
        input_dir=archive_dir.as_posix(),
        output_dir=dm.mha.as_posix(),
        dcm2mha_settings=dm.dcm_settings_json.as_posix(),
    ).convert()