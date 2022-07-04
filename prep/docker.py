from pathlib import Path

import docker

from prep.utils import DirectoryManager


class Dockerfile:
    def __init__(self, dm: DirectoryManager, task_name: str, task_id: int, version: int = 1):
        self._dm = dm
        self.task_name = task_name
        self.task_id = task_id
        self.version = version
        self.tag = f'doduo1.umcn.nl/stan/{task_name}:{version}'

    def output(self, base: str = None) -> str:
        dm = self._dm.from_base(Path(base)) if base else self._dm
        train_commands = '\n'.join([f'RUN nnUNet_train 3d_fullres {self.task_id} {i} --npz' for i in range(5)])

        return f"""FROM python:3.10-slim-bullseye
            
USER user
            
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install nnunet
RUN pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git@more_plotted_details#egg=hiddenlayer

ENV nnUNet_raw_data_base="{dm.nnunet}"
ENV nnUNet_preprocessed="{dm.nnunet_preprocessed}"
ENV RESULTS_FOLDER="{dm.nnunet_results}"

RUN nnUNet_plan_and_preprocess -t {self.task_id} --verify_dataset_integrity
{train_commands}
RUN nnUNet_find_best_configuration -m 3d_fullres -t {self.task_id}"""

    def commands(self) -> str:
        return f"""docker build {self.path.parent.absolute().as_posix()} -t {self.tag}
docker push {self.tag}"""

