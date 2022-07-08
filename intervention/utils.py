import httpx, logging
from pathlib import Path
from datetime import datetime
from typing import Tuple

import gcapi, jsonschema


def now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class DirectoryManager:
    def __init__(self, base: Path, output_dir: Path, task_name: str, task_id: int):
        """
        :param base: Working directory
        :param output_dir: Output directory to store all output in (base / output_dir)
        """
        self._output_dir = Path(output_dir)
        self.output = Path(base / self._output_dir)
        self.mha = self.output / 'mha'
        self.upload = self.output / 'upload'
        self.annotations = self.output / 'annotations'
        self.nnunet = self.output / 'nnUNet_raw_data'

        if not 500 <= task_id < 1000:
            raise ValueError("id must be between 500 and 999")
        self.task_dirname = f'Task{task_id}_{task_name}'
        self._task_dirname = (task_name, task_id)

    def from_base(self, base: Path) -> 'DirectoryManager':
        task_name, task_id = self._task_dirname
        return DirectoryManager(base, self._output_dir, task_name, task_id)


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


def initialize(name: str, schema: dict, kwargs: dict) -> Tuple[Path, DirectoryManager, GCAPI]:
    logging.basicConfig(filename=f'intervention_{name}_{now()}.log',
                        encoding='utf-8',
                        level=logging.INFO)

    jsonschema.validate(kwargs, schema, jsonschema.Draft7Validator)

    archive_dir = Path(kwargs['archive_dir'])
    dm = DirectoryManager(Path('.'), Path(kwargs['out_dir']), kwargs['task_name'], kwargs['task_id'])
    gc = GCAPI(kwargs['gc_slug'], kwargs['gc_api'])

    return archive_dir, dm, gc


settings_schema = {
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
        "prep_run": {
            "description": "select tasks to run, order is non-configurable",
            "type": "array",
            "contains": {
                "type": "string",
                "enum": ["dcm2mha", "upload", "annotate", "mha2nnunet", "all"]
            }
        },
        "results_dir": {
            "description": "segmentation nnUnet containing the fold directories",
            "type": "string"
        }
    },
    "required": ["out_dir", "archive_dir", "gc_slug", "gc_api", "task_name", "task_id"]
}