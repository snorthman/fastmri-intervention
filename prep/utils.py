import shutil, httpx, logging
from pathlib import Path, PurePath
from datetime import datetime

import gcapi


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
        """

        :param base: Working directory
        :param output_dir: Output directory to store all output in (base / output_dir)
        """
        self._output_dir = Path(output_dir)
        self.output = Path(base / self._output_dir)
        self.mha = self.output / 'mha'
        self.upload = self.output / 'upload'
        self.annotations = self.output / 'annotations'
        self.nnunet = self.output / 'nnunet'
        self.nnunet_preprocessed = self.output / 'nnunet_preprocessed'

    @property
    def nnunet_results(self):
        return self.output / f'nnunet_results_{now()}'

    def from_base(self, base: Path) -> 'DirectoryManager':
        return DirectoryManager(base, self._output_dir)


class GCAPI:
    def __init__(self, slug: str, api: str):
        self.client = gcapi.Client(token=api)
        self.slug = slug

        try:
            rs = next(self.client.reader_studies.iterate_all(params={"slug": self.slug}))
        except httpx.HTTPStatusError as e:
            raise ConnectionRefusedError(f'Invalid api key!\n\n{e}')

        self._questions = {v['api_url']: v for v in rs['questions']}
        self._gen_answers = self.client.reader_studies.answers.mine.iterate_all(
            params={"question__reader_study": rs["pk"]})
        self._answers = None
        self._gen_display_sets = self.client.reader_studies.display_sets.iterate_all(
            params={"question__reader_study": rs["pk"]})
        self._display_sets = None
        self._gen_cases = self.client.images.iterate_all(params={"question__reader_study": rs["pk"]})
        self._cases = None

        logging.info('Connected to GC.')

    def image(self, display_set):
        ds = self.display_sets[display_set]
        img = None
        for d in ds['values']:
            if d['interface']['slug'] == 'generic-medical-image':
                img = d['image']
        return self.cases[img]['name']

    @property
    def questions(self):
        return self._questions

    @property
    def answers(self):
        if self._answers is None:
            def gen():
                yield 'raw_answers', list(self._gen_answers)
            get = {name: {v['api_url']: v for v in y} for name, y in gen()}
            self._answers = get['raw_answers']
        return self._answers

    @property
    def display_sets(self):
        if self._display_sets is None:
            def gen():
                yield 'display_sets', list(self._gen_display_sets)
            get = {name: {v['api_url']: v for v in y} for name, y in gen()}
            self._display_sets = get['display_sets']
        return self._display_sets

    @property
    def cases(self):
        if self._cases is None:
            def gen():
                yield 'cases', list(self._gen_cases)
            get = {name: {v['api_url']: v for v in y} for name, y in gen()}
            self._cases = get['cases']
        return self._cases


workflow_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "out_dir": {
            "description": "where all output is sent",
            "type": "string"
        },
        "archive_dir": {
            "description": "where all dicom data is",
            "type": "string"
        },
        "gc_slug": {
            "description": "Grand Challenge reader study slug",
            "type": "string"
        },
        "gc_api": {
            "description": "Grand Challenge API key",
            "type": "string",
            "minLength": 64,
            "maxLength": 64
        },
        "task_name": {
            "description": "for nnUnet",
            "type": "string"
        },
        "task_id": {
            "description": "for nnUnet, between 500 and 999",
            "type": "integer",
            "minimum": 500,
            "maximum": 999
        },
        "docker_version": {
            "type": "integer",
            "minimum": 1
        },
        "run": {
            "description": "select tasks to run, order is non-configurable",
            "type": "array",
            "contains": {
                "type": "string",
                "enum": ["dcm2mha", "upload", "annotate", "mha2nnunet", "docker", "all"]
            }
        }
    },
    "additionalProperties": False
}