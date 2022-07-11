import os, json, concurrent.futures, copy, shutil
from pathlib import Path
from typing import Callable, Dict, Tuple

import click, picai_prep
from tqdm import tqdm
from box import Box
import numpy as np

from intervention.utils import DirectoryManager


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
    options = {'allow_duplicates': True}

    j = dm.output / 'dcm2mha_settings.json'
    with open(j, 'w') as f:
        json.dump({"options": options,
                   "mappings": mappings,
                   "archive": list([a.to_dict() for a in archive])}, f, indent=4)

    return j


def dcm2mha(dm: DirectoryManager, archive_dir: Path, archive_json: Path = None):
    picai_prep.Dicom2MHAConverter(
        input_dir=archive_dir.as_posix(),
        output_dir=dm.mha.as_posix(),
        dcm2mha_settings=archive_json.as_posix(),
    ).convert()


def generate_mha2nnunet_jsons(dm: DirectoryManager) -> Tuple[Path, Path]:
    def walk_mha_archive_add_func(dirpath: Path, filename: str):
        patient_id = dirpath.parts[-1]
        mha = (dm.mha / patient_id / filename)
        annotation = (dm.annotations / filename).with_suffix('.nii.gz')
        fn = filename.split(sep='_')
        if mha.exists() and annotation.exists():
            return {
                "patient_id": patient_id,
                "study_id": f'{fn[1]}_{fn[-1]}'[:-4],
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
    archive = list(archive)

    buckets = {}
    for i, item in enumerate(archive):
        pid, sid = item.patient_id, item.study_id.split('_')[0]
        buckets[pid] = buckets.get(pid, {})
        buckets[pid][sid] = buckets[pid].get(sid, [])
        buckets[pid][sid].append(i)

    splits = [[] for _ in range(min(5 + 1, len(archive)))]
    splits_n = len(splits)
    rng = np.random.default_rng()
    for pid in buckets.keys():
        for sid in buckets[pid].keys():
            # each item with same pid and sid in a bucket
            items: list = buckets[pid][sid]
            while len(items) > 0:
                # select folds with the least items
                splits_c = np.array([len(s) for s in splits])
                S, = np.nonzero(splits_c == splits_c.min(initial=None))
                rng.shuffle(S)
                for s in S[:len(items)]:
                    splits[s].append(archive[items.pop()])

    dataset_json = {
        "description": "Segmentation model for NeedleNet",
        "tensorImageSize": "3D",
        "reference": "",
        "licence": "",
        "release": "0.4",
        "task": dm.task_dirname,
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

    nnunet_split = []
    for S in range(splits_n - 1):
        train, val = [], []
        for s, split in enumerate(splits[:splits_n - 1]):
            group = train if s != S else val
            for item in split:
                group.append(f'{item.patient_id}_{item.study_id}')
        nnunet_split.append({'train': train, 'val': val})

    with open(dm.output / 'nnunet_split.json', 'w') as f:
        json.dump(nnunet_split, f, indent=4)

    dump_settings = lambda A: {"dataset_json": dataset_json,
                               "preprocessing": preprocessing,
                               "archive": list([a.to_dict() for a in A])}

    train = []
    [train.extend(s) for s in splits[:splits_n - 1]]
    with open(train_json := dm.output / f'mha2nnunet_train_settings.json', 'w') as f:
        json.dump(dump_settings(train), f, indent=4)

    test = []
    test.extend(splits[-1])
    with open(test_json := dm.output / f'mha2nnunet_test_settings.json', 'w') as f:
        json.dump(dump_settings(test), f, indent=4)

    return train_json, test_json


def mha2nnunet(dm: DirectoryManager, train_json: Path, test_json: Path):
    train = dm.nnunet / 'train'
    test = dm.nnunet / 'test'
    
    picai_prep.MHA2nnUNetConverter(
        input_path=dm.mha.as_posix(),
        annotations_path=dm.annotations.as_posix(),
        output_path=train.as_posix(),
        settings_path=train_json.as_posix()
    ).convert()

    picai_prep.MHA2nnUNetConverter(
        input_path=dm.mha.as_posix(),
        annotations_path=dm.annotations.as_posix(),
        output_path=test.as_posix(),
        settings_path=test_json.as_posix(),
        out_dir_scans="imagesTs",
        out_dir_annot="labelsTs"
    ).convert()

    train_task = train / dm.task_dirname
    test_task = test / dm.task_dirname
        
    with open(train_task / 'dataset.json') as f:
        train_dataset = json.load(f)
    with open(test_task / 'dataset.json') as f:
        test_dataset = json.load(f)

    dataset = copy.copy(train_dataset)
    dataset['numTest'] = len(test_dataset['training'])
    dataset['test'] = [{'image': i['image'].replace('sTr/', 'sTs/'),
                        'label': i['label'].replace('sTr/', 'sTs/')} for i in test_dataset['training']]

    output = dm.nnunet / dm.task_dirname
    output.mkdir(parents=True)

    shutil.move(train_task / 'imagesTr', output / 'imagesTr')
    shutil.move(train_task / 'labelsTr', output / 'labelsTr')
    shutil.move(test_task / 'imagesTs', output / 'imagesTs')
    shutil.move(test_task / 'labelsTs', output / 'labelsTs')
    with open(output / 'dataset.json', 'w') as f:
        json.dump(dataset, f)

    shutil.rmtree(train)
    shutil.rmtree(test)