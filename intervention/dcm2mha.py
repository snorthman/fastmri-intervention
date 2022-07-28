import picai_prep

from intervention.utils import CommandDCM2MHA


def dcm2mha(cmd: CommandDCM2MHA):
    picai_prep.Dicom2MHAConverter(
        input_dir=cmd.archive_dir.as_posix(),
        output_dir=cmd.out_dir.as_posix(),
        dcm2mha_settings=(cmd.json_dir / 'dcm2mha_settings.json').as_posix(),
    ).convert()