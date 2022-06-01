import os, json, concurrent.futures
from pathlib import Path
from typing import Callable, Dict, Tuple

import click, picai_prep
from tqdm import tqdm
from box import Box


def _walk_archive(in_dir: Path, endswith: str, add_func: Callable[[Path, str], Dict]) -> set:
    archive = set()
    for dirpath, dirnames, filenames in os.walk(in_dir):
        for fn in [f for f in filenames if f.endswith(endswith)]:
            archive.add(Box(add_func(Path(dirpath), fn), frozen_box=True))
    return archive


def generate_dcm2mha_json(in_dir: Path, out_dir: Path) -> Path:
    def walk_dcm_archive(in_dir: Path) -> set:
        return _walk_archive(in_dir, endswith='.dcm',
                      add_func=lambda dp, _: {
                          "patient_id": dp.parts[-3],
                          "study_id": dp.parts[-2].split(sep='.')[-1],
                          "path": dp.as_posix()
                      })

    click.echo(f"Gathering DICOMs from {in_dir} and its subdirectories")
    dirs = list(in_dir.iterdir())

    with concurrent.futures.ThreadPoolExecutor() as executor:
        archives = list(tqdm(executor.map(walk_dcm_archive, dirs), total=len(dirs)))

    archive = set()
    for a in archives:
        archive.update(a)

    mappings = {'needle': {'SeriesDescription': ['naald', 'nld']}}

    j = out_dir / 'dcm2mha_settings.json'
    with open(j, 'w') as f:
        json.dump({"mappings": mappings,
                   "archive": list([a.to_dict() for a in archive])}, f, indent=4)

    return j


def dcm2mha(dcm_dir: Path, out_dir: Path, j: Path = None):
    if not j or not j.exists():
        j = generate_dcm2mha_json(dcm_dir, out_dir)

    picai_prep.Dicom2MHAConverter(
        input_path=dcm_dir.as_posix(),
        output_path=out_dir.as_posix(),
        settings_path=j.as_posix(),
    ).convert()


def generate_mha2nnunet_json(mha_dir: Path, annotations_dir: Path, out_dir: Path) -> Path:
    def walk_mha_archive(in_dir: Path) -> set:
        return _walk_archive(in_dir, endswith='.mha',
                      add_func=lambda dp, fn: {
                        "patient_id": dp.parts[-1],
                        "study_id": fn.split(sep='_')[1],
                        "scan_paths": [(dp / fn).relative_to(mha_dir).as_posix()],
                        "annotation_path": Path(fn).relative_to(annotations_dir).with_suffix('.nii.gz').as_posix()
                    })

    click.echo(f"Gathering MHAs from {mha_dir} and its subdirectories")
    dirs = list(mha_dir.iterdir())
    with concurrent.futures.ThreadPoolExecutor() as executor:
        archives = list(tqdm(executor.map(walk_mha_archive, dirs), total=len(dirs)))

    archive = set()
    for a in archives:
        archive.update(a)

    dataset_json = {
        "description": "",
        "tensorImageSize": "3D",
        "reference": "",
        "licence": "",
        "release": "0.3",
        "modality": {
            "0": "trufi"
        },
        "labels": {
            "0": "background",
            "1": "needle",
            "2": "tip"
        }
    }

    preprocessing = {
        "matrix_size": [
            5,
            256,
            256
        ],
        "spacing": [
            3.0,
            1.094,
            1.094
        ]
    }

    j = out_dir / 'mha2nnunet_settings.json'
    with open(j, 'w') as f:
        json.dump({"dataset_json": dataset_json,
                   "preprocessing": preprocessing,
                   "archive": list([a.to_dict() for a in archive])}, f, indent=4)

    return j


def mha2nnunet(name: str, id: int, mha_dir: Path, annotations_dir: Path, out_dir: Path, j: Path = None):
    if not 500 <= id < 1000:
        raise ValueError("id must be between 500 and 999")

    if not j or not j.exists():
        j = generate_mha2nnunet_json(mha_dir, annotations_dir, out_dir)

    with open(j, 'rw') as f:
        settings = json.load(f)
        settings['dataset_json']['task'] = f"Task{id}_{name}"
        json.dump(settings)

    picai_prep.MHA2nnUNetConverter(
        input_path=mha_dir.as_posix(),
        annotations_path=annotations_dir.as_posix(),
        output_path=out_dir.as_posix(),
        settings_path=j.as_posix()
    ).convert()
