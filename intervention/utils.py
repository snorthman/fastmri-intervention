import httpx, logging, json, copy
from pathlib import Path
from datetime import datetime

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
    def __init__(self, name: str, summary: str, settings: dict):
        self.name = name
        self.summary = summary
        self._settings = settings
        task_name = settings.get('task_name', None)
        task_id = settings.get('task_id', 500)

        self.dm = DirectoryManager(Path('.'), self.validate_dir(settings['out_dir']), task_name, task_id)
        gc_slug = settings.get('gc_slug', None)
        gc_api = settings.get('gc_api', None)
        self.gc = None
        if gc_slug and gc_api:
            self.gc = GCAPI(gc_slug, gc_api)
        self.archive_dir = self.validate_dir(settings.get('archive_dir', None))

        self.trainer = settings.get('trainer', None)
        self.test_percentage = settings.get('test_percentage', 1 / 6)

    def __str__(self):
        return self.name

    @staticmethod
    def validate_dir(path: str):
        if not path:
            return
        path = Path(path)
        if not path.exists() or not path.is_dir():
            raise NotADirectoryError(f'{path} does not exist or is not a directory')
        return path


class Settings:
    def __init__(self, json_path: Path):
        with open(json_path) as f:
            settings = json.load(f)
        self.json = copy.copy(settings)

        n = datetime.now().strftime("%Y%m%d_%H%M%S")
        logging.basicConfig(filename=f'fastmri_intervention_{n}.log',
                            level=logging.INFO)

        base, schemas = Settings._schema()
        jsonschema.validate(settings, base, jsonschema.Draft7Validator)

        props = {}
        for i, cmd in enumerate(settings):
            name = cmd['cmd']
            jsonschema.validate(cmd, schemas[name], jsonschema.Draft7Validator)
            props.update(cmd)
            settings[i] = copy.copy(props)

        self.commands = []
        for cmd in settings:
            name = cmd.pop('cmd')
            properties: dict = schemas[name]['properties']
            schemas[name]['required'] = list(properties.keys())
            jsonschema.validate(cmd, schemas[name], jsonschema.Draft7Validator)

            summary = [schemas[name]['description']]
            for key, val in properties.items():
                desc = val['description']
                summary.append(f'.\t{key}: {desc}\n.\t> {cmd[key]}')

            self.commands.append(Command(name, '\n'.join(summary), cmd))

        logging.info(self.summary())

    def summary(self) -> str:
        return '\n\n'.join([f'({str(i)}) {c.name}: {c.summary}' for i, c in enumerate(self.commands)])

    @staticmethod
    def _schema():
        draft = "http://json-schema.org/draft-07/schema#"

        base = {
            "$schema": draft,
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

        schemas = {}

        out_dir = {
            "description": "where all output is sent",
            "type": "string"
        }
        archive_dir = {
            "description": "where all dicom data is",
            "type": "string"
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

        schemas['dcm'] = object_schema("generate a json settings file for the dcm2mha converter", out_dir=out_dir, archive_dir=archive_dir)
        schemas['dcm2mha'] = object_schema("convert dcm2mha", out_dir=out_dir, archive_dir=archive_dir)
        schemas['upload'] = object_schema("upload MHA to GC", gc_slug=gc_slug, gc_api=gc_api)
        schemas['annotate'] = object_schema("download from GC, then annotate", out_dir=out_dir, gc_slug=gc_slug, gc_api=gc_api)
        schemas['mha2nnunet'] = object_schema("convert dcm2mha", out_dir=out_dir, test_percentage=test_percentage)
        schemas['inference'] = object_schema("inference using trained model", out_dir=out_dir, trainer=trainer)
        schemas['plot'] = object_schema("plot all inferenced directories", out_dir=out_dir)

        return base, schemas