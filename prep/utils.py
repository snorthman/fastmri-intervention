import shutil
from pathlib import Path
from datetime import datetime


def remake_dir(dir: Path) -> Path:
    if dir.exists():
        shutil.rmtree(dir)
    dir.mkdir()
    return dir


def now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def nnunet_dirs(base_dir: Path):
    return base_dir / 'nnunet', base_dir / 'nnunet_preprocessed', base_dir / f'nnunet_results_{now()}'


class DirectoryManager:
    def __init__(self, base: Path, output_dir: Path):
        self._output_dir = Path(output_dir)
        self.output = base / self._output_dir
        self.mha = self.output / 'mha'
        self.annotations = self.output / 'annotations'
        self.nnunet = self.output / 'nnunet'
        self.nnunet_preprocessed = self.output / 'nnunet_preprocessed'

    @property
    def nnunet_results(self):
        return self.output / f'nnunet_results_{now()}'

    def from_base(self, base: Path) -> 'DirectoryManager':
        return DirectoryManager(base, self._output_dir)