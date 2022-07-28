import httpx, logging, json, copy, os
from pathlib import Path
from datetime import datetime
from typing import Callable, Dict, Tuple

from box import Box
import gcapi, jsonschema


def now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def dataset_json(task_dirname: str):
    return {
        "description": "Segmentation model for NeedleNet",
        "tensorImageSize": "3D",
        "reference": "",
        "licence": "",
        "release": "0.4",
        "task": task_dirname,
        "modality": {
            "0": "needle"
        },
        "labels": {
            "0": "background",
            "1": "needle",
            "2": "tip"
        }
    }


def walk_archive(in_dir: Path, endswith: str, add_func: Callable[[Path, str], Dict]) -> set:
    archive = set()
    for dirpath, dirnames, filenames in os.walk(in_dir):
        for fn in [f for f in filenames if f.endswith(endswith)]:
            obj = add_func(Path(dirpath), fn)
            if obj:
                archive.add(Box(obj, frozen_box=True))
    return archive


class DirectoryManager:
    def __init__(self, base: Path, output_dir: Path, task_name: str, task_id: int):
        """
        :param base: Working directory
        :param output_dir: Output directory to store all output in (base / output_dir)
        """
        self._output_dir = Path(output_dir)
        self._base = Path(base)
        self.output = Path(base / self._output_dir)
        self.dcm = self.output / 'dcm'
        self.dcm_settings_json = self.dcm / 'dcm2mha_settings.json'
        self.mha = self.output / 'mha'
        self.upload = self.output / 'upload'
        self.annotations = self.output / 'annotations'
        self.nnunet = self.output / 'nnUNet_raw_data'
        self.nnunet_split_json = self.nnunet / 'nnunet_split.json'
        self.nnunet_train_json = self.nnunet / f'mha2nnunet_train_settings.json'
        self.nnunet_test_json = self.nnunet / f'mha2nnunet_test_settings.json'
        self.results = self.output / 'results'
        self.predict = self.output / 'predict'

        if not 500 <= task_id < 1000:
            raise ValueError("id must be between 500 and 999")
        self.task_dirname = f'Task{task_id}_{task_name}'
        self._task_dirname = (task_name, task_id)

    def refactor(self, base: Path = None, output_dir: Path = None) -> 'DirectoryManager':
        task_name, task_id = self._task_dirname
        output_dir = output_dir if output_dir else self._output_dir
        base = base if base else self._base
        return DirectoryManager(base, output_dir, task_name, task_id)


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


class Command:
    def __init__(self, **kwargs):#name: str, summary: str, base_dir: Path, settings: dict):
        self.name = kwargs['name']
        self.summary = kwargs['summary']
        self._settings = kwargs['settings']
        self._base = kwargs['base_dir']
        self.kwargs = kwargs

    def __str__(self):
        return self.name

    def setup_dir(self, key: str) -> Path:
        new_dir: str = self._settings[key]
        if new_dir.startswith('/'):
            absolute_dir: Path = Path(new_dir)
            if not absolute_dir.exists():
                raise FileExistsError(key)
            if not absolute_dir.is_dir():
                raise NotADirectoryError(key)
            return absolute_dir
        else:
            relative_dir: Path = Path(self._base / new_dir)
            relative_dir.mkdir(exist_ok=True, parents=True)
            return relative_dir


class CommandDCM(Command):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.out_dir = self.setup_dir('out_dir')
        self.archive_dir = self.setup_dir('archive_dir')
        self.mappings = self._settings['mappings']


class CommandDCM2MHA(Command):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.out_dir = self.setup_dir('out_dir')
        self.archive_dir = self.setup_dir('archive_dir')
        self.json_dir = self.setup_dir('json_dir')


class CommandUpload(Command):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        gc_slug: str = self._settings['gc_slug']
        gc_api: str = self._settings['gc_api']
        self.gc = None
        if gc_slug and gc_api:
            self.gc = GCAPI(gc_slug, gc_api)
        else:
            raise AttributeError(f'missing attribute!\ngc_api: {gc_api}\ngc_slug: {gc_slug}')


class CommandAnnotate(CommandUpload):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.out_dir = self.setup_dir('out_dir')
        self.mha_dir = self.setup_dir('mha_dir')


class CommandMHA2nnUNet(Command):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.out_dir = self.setup_dir('out_dir')
        self.mha_dir = self.setup_dir('mha_dir')
        self.annotate_dir = self.setup_dir('annotate_dir')
        self.task_name: str = self._settings['task_name']
        self.task_id: int = self._settings['task_id']
        self.task_dirname = f'Task{self.task_id}_{self.task_name}'
        self.test_percentage: float = self._settings['test_percentage']


class CommandInference(Command):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.out_dir = self.setup_dir('out_dir')
        self.in_dir = self.setup_dir('in_dir')
        self.model_dir = self.setup_dir('model_dir')
        self.trainer: str = self._settings['trainer']


class CommandPlot(Command):
    pass


def _commandFactory(name: str, summary: str, base_dir: Path, settings: dict) -> Command:
    kwargs = {'name': name, 'summary': summary, 'base_dir': base_dir, 'settings': settings}
    if name == 'dcm':
        return CommandDCM(**kwargs)
    if name == 'dcm2mha':
        return CommandDCM2MHA(**kwargs)
    if name == 'upload':
        return CommandUpload(**kwargs)
    if name == 'mha2nnunet':
        return CommandMHA2nnUNet(**kwargs)
    if name == 'annotate':
        return CommandAnnotate(**kwargs)
    if name == 'inference':
        return CommandInference(**kwargs)
    if name == 'plot':
        return CommandPlot(**kwargs)
    raise KeyError(f'unknown name: {name}')


class Settings:
    def __init__(self, json_path: Path):
        with open(json_path) as f:
            settings = json.load(f)
        self.json = settings

        n = datetime.now().strftime("%Y%m%d_%H%M%S")
        logging.basicConfig(filename=f'fastmri_intervention_{n}.log',
                            level=logging.INFO)

        base_schema, schemas = Settings._schema()
        jsonschema.validate(settings, base_schema, jsonschema.Draft7Validator)

        self.base = Path(settings['base_dir'])
        self.base.mkdir(exist_ok=True)

        cmds = settings['commands']

        props = {}
        for i, cmd in enumerate(cmds):
            name = cmd['cmd']
            jsonschema.validate(cmd, schemas[name], jsonschema.Draft7Validator)
            props.update(cmd)
            cmds[i] = copy.copy(props)

        self.commands = []
        for cmd in cmds:
            name = cmd.pop('cmd')
            properties: dict = schemas[name]['properties']
            schemas[name]['required'] = list(properties.keys())
            jsonschema.validate(cmd, schemas[name], jsonschema.Draft7Validator)

            summary = [schemas[name]['description']]
            for key, val in properties.items():
                desc = val['description']
                summary.append(f'.\t{key}: {desc}\n.\t> {cmd[key]}')

            self.commands.append(_commandFactory(name, '\n'.join(summary), self.base, cmd))

        logging.info(self.summary())

    def summary(self) -> str:
        return '\n\n'.join([f'({str(i)}) {c.name}: {c.summary}' for i, c in enumerate(self.commands)])

    @staticmethod
    def _schema():
        draft = "http://json-schema.org/draft-07/schema#"
        base = {
            "$schema": draft,
            "type": "object",
            "properties": {
                "base_dir": {
                    "description": "base directory",
                    "type": "string"
                },
                "commands": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cmd": {
                                "type": "string",
                                "enum": ["dcm", "dcm2mha", "upload", "annotate", "mha2nnunet", "inference", "plot"]
                            }
                        }
                    }
                }
            }
        }

        schemas = {}

        out_dir = {
            "description": "where all output is sent, root is base_dir",
            "type": "string"
        }
        in_dir = {
            "description": "root is base_dir unless it starts with /",
            "type": "string"
        }
        mappings = {
            "description": "picai_prep/dcm2mha mappings",
            "type": "object"
        }
        gc_slug = {
            "description": "Grand Challenge reader study slug",
            "type": "string"
        }
        gc_api = {
            "description": "Grand Challenge API key",
            "type": "string",
            "minLength": 64,
            "maxLength": 64
        }
        task_name = {
            "description": "nnUNet task name",
            "type": "string"
        }
        task_id = {
            "description": "nnUNet task ID",
            "type": "integer",
            "minimum": 500,
            "maximum": 999
        }
        trainer = {
            "description": "model trainer name to inference with",
            "type": "string"
        }
        test_percentage = {
            "description": "mha files to seperate as test set",
            "type": "number",
            "minimum": 0,
            "maximum": 1
        }

        def object_schema(description: str, **properties) -> dict:
            return {
                "$schema": draft,
                "type": "object",
                "description": description,
                "properties": properties
            }

        schemas['dcm'] = object_schema("generate a json settings file for the dcm2mha converter",
                                       archive_dir=in_dir, out_dir=out_dir, mappings=mappings)
        schemas['dcm2mha'] = object_schema("convert dcm2mha",
                                           archive_dir=in_dir, out_dir=out_dir, json_dir=in_dir)
        schemas['upload'] = object_schema("upload MHA to GC",
                                          gc_slug=gc_slug, gc_api=gc_api)
        schemas['annotate'] = object_schema("download from GC, then annotate",
                                            mha_dir=in_dir, out_dir=out_dir,
                                            gc_slug=gc_slug, gc_api=gc_api)
        schemas['mha2nnunet'] = object_schema("convert mha2nnunet",
                                              mha_dir=in_dir, annotate_dir=in_dir, out_dir=out_dir,
                                              test_percentage=test_percentage, task_name=task_name, task_id=task_id)
        schemas['inference'] = object_schema("inference using trained model",
                                             in_dir=in_dir, model_dir=in_dir, out_dir=out_dir,
                                             trainer=trainer)
        schemas['plot'] = object_schema("plot all inferenced directories")

        return base, schemas