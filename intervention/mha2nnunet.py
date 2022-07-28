import json, concurrent.futures, copy, shutil
from pathlib import Path

import click, picai_prep
from tqdm import tqdm
import numpy as np

from intervention.utils import CommandMHA2nnUNet, dataset_json, walk_archive

SPLIT_JSON = 'nnunet_split.json'
TRAIN_JSON = 'mha2nnunet_train_settings.json'
TEST_JSON = 'mha2nnunet_test_settings.json'


def generate_mha2nnunet_jsons(cmd: CommandMHA2nnUNet):
    rng = np.random.default_rng()

    def walk_mha_archive_add_func(dirpath: Path, filename: str):
        patient_id = dirpath.parts[-1]
        mha = (cmd.mha_dir / patient_id / filename)
        annotation = (cmd.annotate_dir / filename).with_suffix('.nii.gz')
        fn = filename.split(sep='_')
        if mha.exists() and annotation.exists():
            return {
                "patient_id": patient_id,
                "study_id": f'{fn[1]}_{fn[-1]}'[:-4],
                "scan_paths": [mha.relative_to(cmd.mha_dir).as_posix()],
                "annotation_path": annotation.relative_to(cmd.annotate_dir).as_posix()
            }

    def walk_mha_archive(in_dir: Path) -> set:
        return walk_archive(in_dir, endswith='.mha', add_func=walk_mha_archive_add_func)

    click.echo(f"Gathering MHAs from {cmd.mha_dir} and its subdirectories")
    dirs = list(cmd.mha_dir.iterdir())
    with concurrent.futures.ThreadPoolExecutor() as executor:
        archives = list(tqdm(executor.map(walk_mha_archive, dirs), total=len(dirs)))

    archive = set()
    for a in archives:
        archive.update(a)
    archive = list(archive)
    rng.shuffle(archive)

    split = max(0, min(len(archive) - 1, round(cmd.test_percentage * len(archive))))
    test_set = archive[:split]
    train_set = archive[split:]

    buckets = {}
    for i, item in enumerate(train_set):
        pid, sid = item.patient_id, item.study_id.split('_')[0]
        buckets[pid] = buckets.get(pid, {})
        buckets[pid][sid] = buckets[pid].get(sid, [])
        buckets[pid][sid].append(i)

    splits = [[] for _ in range(min(5, len(train_set)))]
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
                    splits[s].append(train_set[items.pop()])

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
    for S in range(len(splits)):
        train, val = [], []
        for s, split in enumerate(splits):
            group = train if s != S else val
            for item in split:
                group.append(f'{item.patient_id}_{item.study_id}')
        nnunet_split.append({'train': train, 'val': val})

    with open(cmd.out_dir / SPLIT_JSON, 'w') as f:
        json.dump(nnunet_split, f, indent=4)

    dump_settings = lambda A: {"dataset_json": dataset_json(cmd.task_dirname),
                               "preprocessing": preprocessing,
                               "archive": list([a.to_dict() for a in A])}

    with open(cmd.out_dir / TRAIN_JSON, 'w') as f:
        json.dump(dump_settings([t for s in splits for t in s]), f, indent=4)
    with open(cmd.out_dir / TEST_JSON, 'w') as f:
        json.dump(dump_settings(test_set), f, indent=4)


def mha2nnunet(cmd: CommandMHA2nnUNet):
    train_json = cmd.out_dir / TRAIN_JSON
    test_json = cmd.out_dir / TEST_JSON
    if not train_json.exists() or not test_json.exists():
        generate_mha2nnunet_jsons(cmd)

    train = cmd.out_dir / 'train'
    test = cmd.out_dir / 'test'

    picai_prep.MHA2nnUNetConverter(
        input_path=cmd.mha_dir.as_posix(),
        annotations_path=cmd.annotate_dir.as_posix(),
        output_path=train.as_posix(),
        settings_path=train_json.as_posix()
    ).convert()

    picai_prep.MHA2nnUNetConverter(
        input_path=cmd.mha_dir.as_posix(),
        annotations_path=cmd.annotate_dir.as_posix(),
        output_path=test.as_posix(),
        settings_path=test_json.as_posix(),
        out_dir_scans="imagesTs",
        out_dir_annot="labelsTs"
    ).convert()

    train_task = train / cmd.task_dirname
    test_task = test / cmd.task_dirname

    with open(train_task / 'dataset.json') as f:
        train_dataset = json.load(f)
    with open(test_task / 'dataset.json') as f:
        test_dataset = json.load(f)

    dataset = copy.copy(train_dataset)
    dataset['numTest'] = len(test_dataset['training'])
    dataset['test'] = [{'image': i['image'].replace('sTr/', 'sTs/'),
                        'label': i['label'].replace('sTr/', 'sTs/')} for i in test_dataset['training']]

    output = cmd.out_dir / cmd.task_dirname
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    shutil.move(train_task / 'imagesTr', output / 'imagesTr')
    shutil.move(train_task / 'labelsTr', output / 'labelsTr')
    shutil.move(test_task / 'imagesTs', output / 'imagesTs')
    shutil.move(test_task / 'labelsTs', output / 'labelsTs')
    with open(output / 'dataset.json', 'w') as f:
        json.dump(dataset, f)

    shutil.rmtree(train)
    shutil.rmtree(test)