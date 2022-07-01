import os, json, concurrent.futures
from pathlib import Path
from typing import Callable, Dict

import click, picai_prep
from tqdm import tqdm
from box import Box

from prep.utils import DirectoryManager


def _walk_archive(in_dir: Path, endswith: str, add_func: Callable[[Path, str], Dict]) -> set:
    archive = set()
    for dirpath, dirnames, filenames in os.walk(in_dir):
        for fn in [f for f in filenames if f.endswith(endswith)]:
            obj = add_func(Path(dirpath), fn)
            if obj:
                archive.add(Box(obj, frozen_box=True))
    return archive


def generate_dcm2mha_json(dm: DirectoryManager, archive_dir: Path) -> Path:
    def walk_dcm_archive_add_func(dirpath: Path, _: str):
        return {
            "patient_id": dirpath.parts[-3],
            "study_id": dirpath.parts[-2].split(sep='.')[-1],
            "path": dirpath.as_posix()
        }

    def walk_dcm_archive(in_dir: Path) -> set:
        return _walk_archive(in_dir, endswith='.dcm', add_func=walk_dcm_archive_add_func)

    click.echo(f"Gathering DICOMs from {archive_dir} and its subdirectories")
    dirs = [d.absolute() for d in archive_dir.iterdir()]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        archives = list(tqdm(executor.map(walk_dcm_archive, dirs), total=len(dirs)))

    archive = set()
    for a in archives:
        archive.update(a)

    mappings = {'needle': {'SeriesDescription': ['naald', 'nld']}}
    options = {'random_seed': 0, 'allow_duplicates': True}

    j = dm.output / 'dcm2mha_settings.json'
    with open(j, 'w') as f:
        json.dump({"options": options,
                   "mappings": mappings,
                   "archive": list([a.to_dict() for a in archive])}, f, indent=4)

    return j


def dcm2mha(dm: DirectoryManager, archive_dir: Path, j: Path = None):
    if not j or not j.exists():
        j = generate_dcm2mha_json(dm, archive_dir)

    picai_prep.Dicom2MHAConverter(
        input_dir=archive_dir.as_posix(),
        output_dir=dm.mha.as_posix(),
        dcm2mha_settings=j.as_posix(),
    ).convert()


def generate_mha2nnunet_json(dm: DirectoryManager) -> Path:
    def walk_mha_archive_add_func(dirpath: Path, filename: str):
        patient_id = dirpath.parts[-1]
        mha = (dm.mha / patient_id / filename)
        annotation = (dm.annotations / filename).with_suffix('.nii.gz')
        if mha.exists() and annotation.exists():
            return {
                "patient_id": patient_id,
                "study_id": filename.split(sep='_')[1],
                "scan_paths": [mha.relative_to(dm.mha).as_posix()],
                "annotation_path": annotation.relative_to(dm.annotations).as_posix()
            }

    def walk_mha_archive(in_dir: Path) -> set:
        return _walk_archive(in_dir, endswith='.mha', add_func=walk_mha_archive_add_func)

    click.echo(f"Gathering MHAs from {dm.mha} and its subdirectories")
    dirs = list(dm.mha.iterdir())
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

    j = dm.output / 'mha2nnunet_settings.json'
    with open(j, 'w') as f:
        json.dump({"dataset_json": dataset_json,
                   "preprocessing": preprocessing,
                   "archive": list([a.to_dict() for a in archive])}, f, indent=4)

    return j


def mha2nnunet(dm: DirectoryManager, name: str, id: int, j: Path = None):
    if not 500 <= id < 1000:
        raise ValueError("id must be between 500 and 999")

    if not j or not j.exists():
        j = generate_mha2nnunet_json(dm)

    with open(j, 'r') as f:
        settings = json.load(f)
        settings['dataset_json']['task'] = f"Task{id}_{name}"
    with open(j, 'w') as f:
        json.dump(settings, f)

    picai_prep.MHA2nnUNetConverter(
        input_path=dm.mha.as_posix(),
        annotations_path=dm.annotations.as_posix(),
        output_path=dm.nnunet.as_posix(),
        settings_path=j.as_posix()
    ).convert()
