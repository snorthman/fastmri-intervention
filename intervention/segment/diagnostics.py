import subprocess, os, logging
from pathlib import Path

import torch, nnunet.inference.predict as predict, click

from intervention.utils import Settings


def predict(self, task, trainer="nnUNetTrainerV2", network="3d_fullres",
            checkpoint="model_final_checkpoint", folds="0,1,2,3,4", store_probability_maps=True,
            disable_augmentation=False, disable_patch_overlap=False):
    """
    Use trained nnUNet network to generate segmentation masks
    """

    # Set environment variables
    os.environ['RESULTS_FOLDER'] = str(self.nnunet_results)

    # Run prediction script
    cmd = [
        'nnUNet_predict',
        '-t', task,
        '-i', str(self.nnunet_inp_dir),
        '-o', str(self.nnunet_out_dir),
        '-m', network,
        '-tr', trainer,
        '--num_threads_preprocessing', '2',
        '--num_threads_nifti_save', '1'
    ]

    if folds:
        cmd.append('-f')
        cmd.extend(folds.split(','))

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


def diagnose(settings: Settings):
    s = settings.summary()
    logging.debug(s)
    click.confirm(s, abort=True)

    results_dir = settings.results_dir
