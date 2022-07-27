import os, time, logging, httpx
from pathlib import Path

from tqdm import tqdm

from intervention.utils import GCAPI


def upload_data(input: Path, gc: GCAPI, test: bool = False):
    files = []
    # Loop through files in the specified directory and add their names to the list
    for root, directories, filenames in os.walk(input):
        for direc in directories:
            for file in os.listdir(os.path.join(root, direc)):
                files.append(Path(os.path.join(root, direc, file)))
    if test:
        files = [files[0]]

    total = len(files)
    logging.info(f"Found {total} images (cases) for upload")
    # click.confirm('Confirm to start uploading {total} items', abort=True)

    pks = {}
    for order, file in tqdm(enumerate(files, 1), total=total):
        display_set_pk = gc.client.create_display_sets_from_images(
            reader_study=gc.slug,
            display_sets=[{"generic-medical-image": [file.absolute().as_posix()]}]
        )
        display_set_pk = display_set_pk[0]

        pks[display_set_pk] = {
            "order": order
        }
        logging.info(f'{display_set_pk}: {file.name} ({order})')
        if order % 100 == 0:
            time.sleep(60)

    logging.info("Upload complete ...")
    time.sleep(60)

    logging.info(f"Ordering {total} display sets")
    for pk in tqdm(pks.keys(), total=total):
        gc.client.reader_studies.display_sets.partial_update(pk, **pks[pk])


def delete_all_data(gc: GCAPI):
    display_sets = gc.display_sets
    for display_set in display_sets.values():
        try:
            gc.client.reader_studies.display_sets.delete(display_set['pk'])
        except httpx.HTTPStatusError as e:
            continue


if __name__ == '__main__':
    with open('tests/input/api.txt') as f:
        api_key = f.readline()
    delete_all_data(GCAPI('needle-segmentation-for-interventional-radiology', api_key))

