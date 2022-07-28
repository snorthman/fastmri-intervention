import picai_prep

from intervention.utils import Command


def dcm2mha(cmd: Command):
    cmd.assert_attributes('out_dir', 'archive_dir')

    picai_prep.Dicom2MHAConverter(
        input_dir=cmd.archive_dir.as_posix(),
        output_dir=cmd.out_dir.as_posix(),
        dcm2mha_settings=dm.dcm_settings_json.as_posix(),
    ).convert()