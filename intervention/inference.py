# import os, logging, json, shutil
# import subprocess
# from typing import List
# from pathlib import Path
#
# from picai_prep import MHA2nnUNetConverter
# import SimpleITK as sitk
# import matplotlib.pyplot as plt
# import numpy as np
#
# from intervention.utils import dataset_json, CommandInference, CommandDCM, CommandDCM2MHA, CommandMHA2nnUNet
# from intervention.dcm import generate_dcm2mha_json
# from intervention.dcm2mha import dcm2mha
#
#
# def nnUNet_predict(results_dir: Path, input_dir: Path, output_dir: Path, task: str, trainer: str, folds: List = None,
#                     network: str = "3d_fullres", checkpoint: str = "model_final_checkpoint",
#                     store_probability_maps: bool = True, disable_augmentation: bool = False,
#                     disable_patch_overlap: bool = False):
#     """
#     Use trained nnUNet network to generate segmentation masks
#     """
#
#     # Set environment variables
#     os.environ['RESULTS_FOLDER'] = str(results_dir)
#
#     # Run prediction script
#     cmd = [
#         'nnUNet_predict',
#         '-t', task,
#         '-i', str(input_dir),
#         '-o', str(output_dir),
#         '-m', network,
#         '-tr', trainer if trainer else "nnUNetTrainerV2",
#         '--num_threads_preprocessing', '2',
#         '--num_threads_nifti_save', '1'
#     ]
#
#     # if folds:
#     #     cmd.append('-f')
#     #     cmd.extend(folds.split(','))
#
#     if checkpoint:
#         cmd.append('-chk')
#         cmd.append(checkpoint)
#
#     if store_probability_maps:
#         cmd.append('--save_npz')
#
#     if disable_augmentation:
#         cmd.append('--disable_tta')
#
#     if disable_patch_overlap:
#         cmd.extend(['--step_size', '1'])
#
#     try:
#         subprocess.check_call(cmd)
#     except Exception as e:
#         print(' '.join(cmd))
#         logging.info(' '.join(cmd))
#         logging.info(str(e))
#
#
# def get_pid_sid(file: Path):
#     img = sitk.ReadImage(file.as_posix())
#     pid = img.GetMetaData('0010|0020').strip()
#     sid = img.GetMetaData('0020|000d').strip().split('.')[-1]
#     return pid, sid
#
#
# def inference(cmd: CommandInference):
#     # convert files found in /predict to nii.gz
#
#     # dcm2mha
#     dcm_out_dir = str(cmd.in_dir / 'dcm')
#     dcm_archive_dir = str(cmd.in_dir)
#     dcm_mappings = {'mappings': {'auto': {'Modality': ['MR']}}}
#     dcm_cmd = CommandDCM(out_dir=dcm_out_dir, archive_dir=dcm_archive_dir, mappings=dcm_mappings, **cmd.kwargs)
#     generate_dcm2mha_json(dcm_cmd)
#
#     dcm2mha_out_dir = str(cmd.in_dir / 'mha')
#     dcm2mha_cmd = CommandDCM2MHA(out_dir=dcm2mha_out_dir, archive_dir=dcm_archive_dir, json_dir=dcm_out_dir, **cmd.kwargs)
#     dcm2mha(dcm2mha_cmd)
#
#     # mha2nnunet
#     mha2nnunet_out_dir = str(cmd.in_dir / 'dcm')
#     mha2nnunet_mha_dir
#     annotate_dir
#     # self.out_dir = self.setup_dir('out_dir')
#     # self.mha_dir = self.setup_dir('mha_dir')
#     # self.annotate_dir = self.setup_dir('annotate_dir')
#     # self.task_name: str = self._settings['task_name']
#     # self.task_id: int = self._settings['task_id']
#     # self.task_dirname = f'Task{self.task_id}_{self.task_name}'
#     # self.test_percentage: float = self._settings['test_percentage']
#     inference_dm.mha.mkdir(exist_ok=True)
#     mha2nnunet_archive = []
#     for directory in [dm.predict] + list(inference_dm.mha.iterdir()):
#         for path in directory.iterdir():
#             if path.suffix == '.mha':
#                 pid, sid = get_pid_sid(path)
#                 pid_dir = inference_dm.mha / pid
#                 pid_dir.mkdir(exist_ok=True)
#                 mha = pid_dir / path.name
#                 if not mha.exists():
#                     shutil.copyfile(path, pid_dir / path.name)
#
#                 item = {
#                     'patient_id': pid,
#                     'study_id': sid,
#                     'scan_paths': [path.absolute().as_posix()]
#                 }
#                 annotation = dm.annotations / path.with_suffix('.nii.gz').name
#                 if annotation.exists():
#                     item['annotation_path'] = annotation.absolute().as_posix()
#                 mha2nnunet_archive.append(item)
#     mha2nnunet_settings = {
#         "archive": mha2nnunet_archive,
#         "dataset_json": dataset_json(dm.task_dirname),
#         "preprocessing": {
#             "spacing": [
#                 3.0,
#                 1.094,
#                 1.094
#             ]
#         }
#     }
#
#     inference_dm.nnunet.mkdir(exist_ok=True)
#     mha2nnunet_settings_json = inference_dm.nnunet / 'mha2nnunet_settings.json'
#     with open(mha2nnunet_settings_json, 'w') as f:
#         json.dump(mha2nnunet_settings, f, indent=4)
#
#     MHA2nnUNetConverter(
#         input_path=inference_dm.mha.as_posix(),
#         output_path=inference_dm.nnunet.as_posix(),
#         settings_path=mha2nnunet_settings_json.as_posix(),
#         out_dir_scans='images',
#         out_dir_annot='labels'
#     ).convert()
#
#     # clean up
#     nnunet_dir = inference_dm.nnunet / inference_dm.task_dirname
#     (nnunet_dir / 'labels').mkdir(exist_ok=True)
#     for tr in [nnunet_dir / 'images', nnunet_dir / 'labels']:
#         shutil.move(tr, inference_dm.output)
#
#     shutil.rmtree(inference_dm.dcm)
#     shutil.rmtree(inference_dm.mha)
#     shutil.rmtree(inference_dm.nnunet)
#
#     nnUNet_predict(dm.output / 'results', inference_dm.output / 'images', inference_dm.output, dm.task_dirname,
#                    checkpoint='model_best', trainer=trainer)
#
#     return inference_dm.output
#
#
# class Prediction:
#     def __init__(self, path: Path, image_dir: Path, label_dir: Path):
#         self.prediction = path
#         self.name = path.name.split('.')[0]
#         self.image = self._find(image_dir, self.name)
#         self.label = self._find(label_dir, self.name)
#         logging.info(f'path: {path}, image: {self.image}, label: {self.label}')
#
#     @staticmethod
#     def _find(in_dir: Path, name: str) -> Path:
#         for item in in_dir.iterdir():
#             if item.name.startswith(name):
#                 return item
#
#     @staticmethod
#     def _read_image(path: Path):
#         if not path:
#             return np.array([1])
#
#         img = sitk.ReadImage(path.as_posix())
#         img_nda = sitk.GetArrayViewFromImage(img)
#         img_nda = img_nda[img_nda.shape[0] // 2, :, :]
#         img_max = img_nda.max()
#         return img_nda / img_max if img_max > 0 else img_nda
#
#     def set_axes(self, axes: np.ndarray):
#         # image, image with predict, image with annotation (optional)
#         image = self._read_image(self.image)
#         prediction = self._read_image(self.prediction)
#         prediction = np.where(prediction > 0, prediction, image)
#         label = self._read_image(self.label)
#         label = np.where(label > 0, label, image)
#
#         for i, (img, title) in enumerate([(image, 'scan'), (prediction, 'prediction'), (label, 'annotation')]):
#             axes[0, i].imshow(img, interpolation=None, cmap='gray')
#             axes[0, i].set_title(title)
#
#
# def plot(dm: DirectoryManager):
#     for d in dm.predict.iterdir():
#         if d.is_dir() and d.name.startswith('inference_'):
#             inference_dir = d
#
#             predictions = []
#             for path in inference_dir.iterdir():
#                 if path.name.endswith('.nii.gz'):
#                     predictions.append(Prediction(path, inference_dir / 'images', inference_dir / 'labels'))
#
#             if len(predictions) == 0:
#                 logging.error('no predictions found!')
#                 return
#
#             for prediction in predictions:
#                 try:
#                     f, axes = plt.subplots(1, 3)
#                     axes = axes.reshape(1, 3)
#                     prediction.set_axes(axes)
#                     plt.setp([a.get_yticklabels() for a in axes[:, 1:].flatten()], visible=False)
#                     plt.suptitle(prediction.name)
#                     plt.savefig(inference_dir / f'plot_{prediction.name}.png',
#                                 dpi=180, transparent=True, bbox_inches='tight', pad_inches=0)
#                 except Exception as e:
#                     logging.error(str(e))
#                     print(str(e))
