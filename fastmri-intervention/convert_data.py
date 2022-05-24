import os, json
from pathlib import Path

import click, picai_prep
from p_tqdm import p_uimap
from box import Box


def generate_dcm2mha_json(input: Path, output: Path) -> Path:
    click.echo(f"[prepare_data].\n")

    click.echo(f"Gathering DICOMs from {input} and its subdirectories")
    dirs = list(input.iterdir())
    archives = p_uimap(_walk_archive, dirs)
    archive = set()
    for a in archives:
        archive.update(a)

    with open(output := (output / 'dcm2mha_settings.json'), 'w') as f:
        json.dump({"mappings": {'needle': {'SeriesDescription': ['naald', 'nld']}},
                   "archive": list([a.to_dict() for a in archive])}, f, indent=4)

    return output


def _walk_archive(input: Path) -> set:
    archive = set()
    for dirpath, dirnames, filenames in os.walk(input):
        for _ in [f for f in filenames if f.endswith(".dcm")]:
            dp = Path(dirpath)
            archive.add(Box({
                "patient_id": dp.parts[-3],
                "study_id": dp.parts[-2],
                "path": dp.as_posix()
            }, frozen_box=True))
    return archive


def dcm2mha(dcm_dir: Path, mha_dir: Path, json: Path = None):
    if not json or not json.exists():
        json = generate_dcm2mha_json(dcm_dir, mha_dir)

    converter = picai_prep.Dicom2MHAConverter(
        input_path=dcm_dir.as_posix(),
        output_path=mha_dir.as_posix(),
        settings_path=json.as_posix(),
    )
    converter.convert()