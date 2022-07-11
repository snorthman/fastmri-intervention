import subprocess, os, logging
from typing import List
from pathlib import Path

import click

from intervention.utils import Settings, DirectoryManager


def _nnUNet_predict(dm: DirectoryManager, results_dir: Path, folds: List = None, trainer: str = "nnUNetTrainerV2",
                    network: str = "3d_fullres", checkpoint: str = "model_final_checkpoint",
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
        '-t', dm.task_dirname,
        '-i', str(dm.nnunet / dm.task_dirname / 'imagesTs'),
        '-o', str(dm.predict),
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
    s = settings.summary()
    logging.debug(s)
    click.confirm(s, abort=True)

    _nnUNet_predict(settings.dm, settings.results_dir, checkpoint='model_best')
