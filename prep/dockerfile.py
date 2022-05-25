from prep.utils import DirectoryManager


class Dockerfile:
    def __init__(self, dm: DirectoryManager, task_name: str, task_id: int, version: int = 1):
        dm_sol = dm.from_base('/mnt/netcache/pelvis')
        train_commands = '\n'.join([f'RUN nnUNet_train 3d_fullres {task_id} {i} --npz' for i in range(5)])
        with open(dockerfile := (dm.output / 'Dockerfile'), 'w') as d:
            d.write(f"""FROM python:3.10-slim-bullseye
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install nnunet
RUN pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git@more_plotted_details#egg=hiddenlayer

ENV nnUNet_raw_data_base="{dm_sol.nnunet}"
ENV nnUNet_preprocessed="{dm_sol.nnunet_preprocessed}"
ENV RESULTS_FOLDER="{dm_sol.nnunet_results}"

RUN nnUNet_plan_and_preprocess -t {task_id} --verify_dataset_integrity
{train_commands}
RUN nnUNet_find_best_configuration -m 3d_fullres -t {task_id}
            """)
        self.path = dockerfile
        self.tag = f'doduo1.umcn.nl/stan/{task_name}:{version}'

    def commands(self) -> str:
        return f"""docker build {self.path.parent.absolute().as_posix()} -t {self.tag}
docker push {self.tag}
        """
