import json, concurrent.futures
from pathlib import Path

import click
from tqdm import tqdm

from intervention.utils import DirectoryManager, walk_archive


def generate_dcm2mha_json(dm: DirectoryManager, archive_dir: Path) -> Path:
    dcm2mha_settings = dm.dcm / 'dcm2mha_settings.json'

    def walk_dcm_archive_add_func(dirpath: Path, _: str):
        return {
            "patient_id": dirpath.parts[-3],
            "study_id": dirpath.parts[-2].split(sep='.')[-1],
            "path": dirpath.as_posix()
        }

    def walk_dcm_archive(in_dir: Path) -> set:
        return walk_archive(in_dir, endswith='.dcm', add_func=walk_dcm_archive_add_func)

    click.echo(f"Gathering DICOMs from {archive_dir} and its subdirectories")
    dirs = [d.absolute() for d in archive_dir.iterdir()]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        archives = list(tqdm(executor.map(walk_dcm_archive, dirs), total=len(dirs)))

    archive = set()
    for a in archives:
        archive.update(a)

    mappings = {'needle': {'SeriesDescription': ['naald', 'nld']}}
    options = {'allow_duplicates': True}

    dm.dcm.mkdir(exist_ok=True)
    with open(dcm2mha_settings, 'w') as f:
        json.dump({"options": options,
                   "mappings": mappings,
                   "archive": list([a.to_dict() for a in archive])}, f, indent=4)

    return dcm2mha_settings