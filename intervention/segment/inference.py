import os, logging, json, shutil
import subprocess
from typing import List
from pathlib import Path

from picai_prep import MHA2nnUNetConverter
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

from intervention.utils import Settings, now, dataset_json
from intervention.prep.convert import dcm2mha


def nnUNet_predict(results_dir: Path, input_dir: Path, output_dir: Path, task: str, trainer: str, folds: List = None,
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
        '-t', task,
        '-i', str(input_dir),
        '-o', str(output_dir),
        '-m', network,
        '-tr', trainer if trainer else "nnUNetTrainerV2",
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

    try:
        subprocess.check_call(cmd)
    except:
        print(' '.join(cmd))


def get_pid_sid(file: Path):
    img = sitk.ReadImage(file.as_posix())
    pid = img.GetMetaData('0010|0020').strip()
    sid = img.GetMetaData('0020|000d').strip().split('.')[-1]
    return pid, sid


def inference(settings: Settings):
    print(settings.summary())
    # convert files found in /predict to nii.gz

    inference_dm = settings.dm.refactor(output_dir=settings.dm.predict / f'inference_{now()}')
    inference_dm.output.mkdir(parents=True, exist_ok=False)

    # dcm2mha
    dcm2mha_archive = []
    for path in settings.dm.predict.iterdir():
        if path.is_dir():
            for file in path.iterdir():
                if file.suffix == '.dcm':
                    pid, sid = get_pid_sid(file)
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
                pid, sid = get_pid_sid(path)
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
    (nnunet_dir / 'labels').mkdir(exist_ok=True)
    for tr in [nnunet_dir / 'images', nnunet_dir / 'labels']:
        shutil.move(tr, inference_dm.output)

    shutil.rmtree(inference_dm.dcm)
    shutil.rmtree(inference_dm.mha)
    shutil.rmtree(inference_dm.nnunet)

    nnUNet_predict(settings.dm.output / 'results', inference_dm.output / 'images', inference_dm.output, settings.dm.task_dirname,
                    checkpoint='model_best', trainer=settings.trainer)

    print('Inference complete.')


class Prediction:
    def __init__(self, path: Path, image_dir: Path, label_dir: Path):
        self.prediction = path
        name = path.name.split('.')[0]
        self.image = self._find(image_dir, name)
        self.label = self._find(label_dir, name)

    @staticmethod
    def _find(in_dir: Path, name: str):
        for item in in_dir.iterdir():
            if item.name.startswith(name):
                return item

    @staticmethod
    def _read_image(path: Path):
        if not path:
            return np.array([0.5])
        img = sitk.ReadImage(path.as_posix())
        img_nda = sitk.GetArrayViewFromImage(img)
        img_nda = img_nda[img_nda.shape[0] // 2, :, :]
        img_max = img_nda.max()
        return img_nda / img_max if img_max > 0 else img_nda

    def set_axes(self, axes: np.ndarray, row: int):
        # image, image with predict, image with annotation (optional)

        image = self._read_image(self.image)
        prediction = self._read_image(self.prediction)
        prediction = np.where(prediction > 0, prediction, image)
        label = self._read_image(self.label)
        label = np.where(label > 0, label, image)

        for i, (img, title) in enumerate([(image, 'Scan'), (prediction, 'Prediction'), (label, 'Annotation')]):
            axes[row, i].imshow(img, interpolation=None, cmap='gray')
            axes[row, i].set_title(title)


def plot(inference_dir: Path):
    if not inference_dir.is_dir():
        raise NotADirectoryError()

    predictions = []
    for path in inference_dir.iterdir():
        if path.name.endswith('.nii.gz'):
            predictions.append(Prediction(path, inference_dir / 'images', inference_dir / 'labels'))

    rows = len(predictions)
    f, axes = plt.subplots(rows, 3)
    if axes.ndim == 1:
        axes = axes.reshape(rows, 3)
    for i, prediction in enumerate(predictions):
        try:
            prediction.set_axes(axes, i)
        except Exception as e:
            print(str(e))

    plt.setp([a.get_yticklabels() for a in axes[:, 1:].flatten()], visible=False)
    f.subplots_adjust(hspace=0.3)

    plt.savefig(inference_dir / f'{inference_dir.name}.png', dpi=180)
    plt.show()


if __name__ == '__main__':
    predict_dir = Path('tests/input/predict_results')
    plot(predict_dir)