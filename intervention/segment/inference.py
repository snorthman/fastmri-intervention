import subprocess, os, logging, json, shutil
from typing import List
from pathlib import Path

import click
from picai_prep import MHA2nnUNetConverter

from intervention.utils import Settings, DirectoryManager, now, dataset_json
from intervention.prep.convert import dcm2mha, mha2nnunet


def _nnUNet_predict(results_dir: Path, input_dir: Path, output_dir: Path, task: str, folds: List = None,
                    trainer: str = "nnUNetTrainerV2", network: str = "3d_fullres", checkpoint: str = "model_final_checkpoint",
                    store_probability_maps: bool = True, disable_augmentation: bool = False,
                    disable_patch_overlap: bool = False):
    """
    Use trained nnUNet network to generate segmentation masks
    """

    # Set environment variables
    os.environ['RESULTS_FOLDER'] = str(results_dir)

    # Run prediction script
    cmd = [
        'nnUNet_predict',
        '-t', task,
        '-i', str(input_dir),
        '-o', str(output_dir),
        '-m', network,
        '-tr', trainer,
        '--num_threads_preprocessing', '2',
        '--num_threads_nifti_save', '1'
    ]

    # if folds:
    #     cmd.append('-f')
    #     cmd.extend(folds.split(','))

    if checkpoint:
        cmd.append('-chk')
        cmd.append(checkpoint)

    if store_probability_maps:
        cmd.append('--save_npz')

    if disable_augmentation:
        cmd.append('--disable_tta')

    if disable_patch_overlap:
        cmd.extend(['--step_size', '1'])

    subprocess.check_call(cmd)


def inference(settings: Settings):
    s = settings.summary()
    logging.debug(s)
    # click.confirm(s, abort=True)

    # convert files found in /predict to nii.gz

    inference_dm = settings.dm.refactor(output_dir=settings.dm.predict / f'inference_{now()}')
    inference_dm.output.mkdir(parents=True, exist_ok=False)

    # dcm2mha
    dcm2mha_archive = []
    for path in settings.dm.predict.iterdir():
        if path.is_dir():
            for file in path.iterdir():
                if file.suffix == '.dcm':
                    pid, sid = tuple(path.name.split('_'))
                    dcm2mha_archive.append({
                        'patient_id': pid,
                        'study_id': sid,
                        'path': path.absolute().as_posix()
                    })
                    break

    dcm2mha_settings = {'options': {'allow_duplicates': True},
                        'mappings': {'needle': {'Modality': ['MR']}},
                        'archive': dcm2mha_archive}

    inference_dm.dcm.mkdir(exist_ok=True)
    with open(inference_dm.dcm / 'dcm2mha_settings.json', 'w') as f:
        json.dump(dcm2mha_settings, f, indent=4)
    dcm2mha(inference_dm, settings.dm.predict)

    # mha2nnunet
    inference_dm.mha.mkdir(exist_ok=True)
    mha2nnunet_archive = []
    for directory in [settings.dm.predict] + list(inference_dm.mha.iterdir()):
        for path in directory.iterdir():
            if path.suffix == '.mha':
                pid, sid, _ = tuple(path.name.split('_', maxsplit=2))
                pid_dir = inference_dm.mha / pid
                pid_dir.mkdir(exist_ok=True)
                mha = pid_dir / path.name
                if not mha.exists():
                    shutil.copyfile(path, pid_dir / path.name)

                item = {
                    'patient_id': pid,
                    'study_id': sid,
                    'scan_paths': [path.absolute().as_posix()]
                }
                annotation = settings.dm.annotations / path.with_suffix('.nii.gz').name
                if annotation.exists():
                    item['annotation_path'] = annotation.absolute().as_posix()
                mha2nnunet_archive.append(item)
    mha2nnunet_settings = {
        "archive": mha2nnunet_archive,
        "dataset_json": dataset_json(settings.dm.task_dirname),
        "preprocessing": {
            "spacing": [
                3.0,
                1.094,
                1.094
            ]
        }
    }

    inference_dm.nnunet.mkdir(exist_ok=True)
    mha2nnunet_settings_json = inference_dm.nnunet / 'mha2nnunet_settings.json'
    with open(mha2nnunet_settings_json, 'w') as f:
        json.dump(mha2nnunet_settings, f, indent=4)

    MHA2nnUNetConverter(
        input_path=inference_dm.mha.as_posix(),
        output_path=inference_dm.nnunet.as_posix(),
        settings_path=mha2nnunet_settings_json.as_posix(),
        out_dir_scans='images',
        out_dir_annot='labels'
    ).convert()

    # clean up
    nnunet_dir = inference_dm.nnunet / inference_dm.task_dirname
    for tr in [nnunet_dir / 'images', nnunet_dir / 'labels']:
        shutil.move(tr, inference_dm.output)

    shutil.rmtree(inference_dm.dcm)
    shutil.rmtree(inference_dm.mha)
    shutil.rmtree(inference_dm.nnunet)

    _nnUNet_predict(settings.dm.output / 'results', inference_dm.output / 'images', inference_dm.output, settings.dm.task_dirname,
                    checkpoint='model_best', trainer=settings.trainer)

    print('done ')


