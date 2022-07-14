import subprocess, os, logging, shutil
from typing import List
from pathlib import Path

import click

from intervention.utils import Settings, DirectoryManager, now


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


def predict(settings: Settings):
    input_dir = settings.dm.predict / 'input'
    if not input_dir.exists():
        raise NotADirectoryError(f'expected {str(input_dir)} with nii.gz files, nothing to predict')

    s = settings.summary()
    logging.debug(s)
    click.confirm(s, abort=True)

    output_dir = settings.dm.predict / f'segment_{now()}'
    shutil.move(input_dir, input_dir := output_dir / 'input')

    _nnUNet_predict(settings.dm.output / 'results', input_dir, output_dir, settings.dm.task_dirname, checkpoint='model_best')
