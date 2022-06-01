import os, threading
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List
from datetime import datetime

import click, gcapi, numpy as np, SimpleITK as sitk
from shapely.geometry import Polygon, Point, LineString, MultiPoint
from quaternion import from_vector_part, rotate_vectors
from tqdm import tqdm

# in mm
diameter_base = 6
diameter_needle = 3


@dataclass
class Annotation:
    x: int
    y: int
    z: int

    def is_zero(self):
        return self.x == 0 and self.y == 0 and self.z == 0

    def as_tuple(self):
        return self.x, self.y, self.z

    def as_ndarray(self):
        return np.array([self.x, self.y, self.z])


@dataclass
class Answer:
    name: str = 'untitled'
    mha: Path = None
    base: Annotation = Annotation(0, 0, 0)
    needle: Annotation = Annotation(0, 0, 0)
    tip: Annotation = Annotation(0, 0, 0)
    no_needle: bool = True
    _error: str = None

    # how to deal with more creators (snorthman)?

    def _get_error(self) -> str:
        return f'{self.name}: ' + self._error if self._error else 'Annotated'

    def _set_error(self, error):
        self._error = f'{type(error).__name__}({str(error)}'

    error = property(fget=_get_error, fset=_set_error)

    def is_valid(self) -> bool:
        return self.mha is not None and \
               self.mha.exists() and \
               not all(p.is_zero() for p in [self.base, self.needle, self.tip]) and \
               not self.no_needle and \
               self._error is None


class Boundary:
    def __init__(self, a: np.ndarray, b: np.ndarray, label: int, thickness: float = 1):
        self.label = label

        thickness /= 2
        Q = from_vector_part((a - b) / np.linalg.norm(a - b))

        # https://i.stack.imgur.com/iizbo.jpg
        P1 = a + rotate_vectors(Q, [-1, 0, 0]) * thickness
        P2 = a + rotate_vectors(Q, [0, 1, 0]) * thickness
        P3 = a + rotate_vectors(Q, [1, 0, 0]) * thickness
        P4 = a + rotate_vectors(Q, [0, -1, 0]) * thickness
        P5 = b + rotate_vectors(Q, [-1, 0, 0]) * thickness
        P6 = b + rotate_vectors(Q, [0, 1, 0]) * thickness
        P7 = b + rotate_vectors(Q, [1, 0, 0]) * thickness
        P8 = b + rotate_vectors(Q, [0, -1, 0]) * thickness
        P = [P1, P2, P3, P4, P5, P6, P7, P8]
        self._center = np.array(LineString([P1, P7]).interpolate(0.5, normalized=True).coords).flatten()
        self._XY = Polygon(MultiPoint([p[:2] for p in P]).convex_hull.boundary)
        self._YZ = Polygon(MultiPoint([p[1:] for p in P]).convex_hull.boundary)

    @property
    def center(self) -> np.ndarray:
        return self._center

    def contains(self, p: np.ndarray) -> bool:
        xy, yz = Point(p[:2]), Point(p[1:])
        return self._XY.contains(xy) and self._YZ.contains(yz)


def _get_answers(mha_dir: Path, slug: str, api: str) -> List[Answer]:
    mha = dict()
    for root, dirs, files in os.walk(mha_dir):
        for file in files:
            if file.endswith('.mha'):
                mha[file] = Path(root) / file

    client = gcapi.Client(token=api)
    rs = next(client.reader_studies.iterate_all(params={"slug": slug}))

    gen_raw_answers = client.reader_studies.answers.mine.iterate_all(params={"question__reader_study": rs["pk"]})
    gen_display_sets = client.reader_studies.display_sets.iterate_all(params={"question__reader_study": rs["pk"]})
    gen_cases = client.images.iterate_all(params={"question__reader_study": rs["pk"]})

    def gen():
        for name, g in [('raw_answers', gen_raw_answers), ('display_sets', gen_display_sets), ('cases', gen_cases)]:
            yield name, list(g)

    get = {name: {v['api_url']: v for v in y} for name, y in gen()}

    raw_questions = {v['api_url']: v for v in rs['questions']}
    raw_answers = get['raw_answers']
    display_sets = get['display_sets']
    cases = get['cases']

    list_to_annotation = lambda array: Annotation(x=array[0], y=array[1], z=array[2])

    answers = dict()
    for url, ra in raw_answers.items():
        if (display_set := ra['display_set']) in answers:
            a = answers[display_set]
        else:
            answers[display_set] = (a := Answer())

        if not (question := raw_questions.get(ra['question'])):
            click.echo(f'unknown question: {ra["question"]}')
            continue
        question = question['question_text']
        if question == 'No needle':
            a.no_needle = ra['answer']
        elif question == 'Base':
            a.base = list_to_annotation(ra['answer']['point'])
        elif question == 'Needle':
            a.needle = list_to_annotation(ra['answer']['point'])
        elif question == 'Tip':
            a.tip = list_to_annotation(ra['answer']['point'])

        try:
            ds = display_sets[display_set]
            img = None
            for d in ds['values']:
                if d['interface']['slug'] == 'generic-medical-image':
                    img = d['image']
            a.name = cases[img]['name']
            a.mha = mha[a.name]
        except Exception as e:
            a.error = FileNotFoundError(str(e))
            continue

    return [a for a in answers.values()]


def write_annotations(mha_dir: Path, out_dir: Path, slug: str, api: str, base_needle: int = 1, needle_tip: int = 2):
    context = threading.local()

    if not all(0 < x < 3 for x in [base_needle, needle_tip]):
        raise ValueError("base_needle and needle_tip must be 1 and/or 2")

    def initializer_worker():
        context.ifr = sitk.ImageFileReader()

    def _write_annotation(answer: Answer) -> bool:
        if not answer.is_valid():
            return False

        try:
            context.ifr.SetFileName(str(answer.mha.absolute()))
            context.ifr.ReadImageInformation()
            mha: sitk.Image = context.ifr.Execute()

            (sz := list(mha.GetSize())).reverse()
            annotation = sitk.GetImageFromArray(np.zeros(sz))
            annotation.SetDirection(mha.GetDirection())
            annotation.SetOrigin(mha.GetOrigin())
            annotation.SetSpacing(mha.GetSpacing())
            [annotation.SetMetaData(k, mha.GetMetaData(k)) for k in mha.GetMetaDataKeys()]

            base, needle, tip = (p.as_ndarray() for p in [answer.base, answer.needle, answer.tip])

            boundary_base_needle = Boundary(base, needle, base_needle, thickness=diameter_base)
            boundary_needle_tip = Boundary(needle, tip, needle_tip, thickness=diameter_needle)

            for boundary in [boundary_base_needle, boundary_needle_tip]:
                trail = [start := mha.TransformPhysicalPointToIndex(boundary.center)]
                explored = {start}

                while len(trail) > 0:
                    X, Y, Z = trail.pop()
                    try:
                        annotation.SetPixel(X, Y, Z, boundary.label)
                    except Exception:
                        pass

                    group = []
                    for x in [X - 1, X + 1]:
                        group.append((x, Y, Z))
                    for y in [Y - 1, Y + 1]:
                        group.append((X, y, Z))
                    for z in [Z - 1, Z + 1]:
                        group.append((X, Y, z))
                    group = [g for g in group if g not in explored]

                    trail += [g for g in group
                              if boundary.contains(np.array(mha.TransformIndexToPhysicalPoint(g)))]
                    explored = explored.union(group)

            sitk.WriteImage(annotation, fileName=str(out_dir / answer.mha.with_suffix('.nii.gz').name), useCompression=True)
            return True
        except Exception as e:
            answer.error = e
            return False

    click.echo(f'\ncreating annotations in\n\t{out_dir}\nusing\n\t{mha_dir}')

    answers = _get_answers(mha_dir, slug, api)

    click.echo(f'downloaded {len(answers)} answers from grand challenge')

    successes, errors = 0, 0
    with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4), initializer=initializer_worker) as pool:
        futures = {pool.submit(_write_annotation, a): a for a in answers}
        for future in tqdm(as_completed(futures), total=len(answers)):
            try:
                successes += 1 if future.result() else 0
            except Exception as e:
                click.echo(f'Unexpected error: {e}')
                errors += 1
    skips = len(answers) - successes - errors
    click.echo(f'wrote {successes} annotations, with {skips} skipped and {errors} failed')

    with open(out_dir / f'{datetime.now().strftime("%Y%m%d%H%M%S")}.log', 'w') as f:
        f.writelines([f'{a.error}\n' for a in answers if not a.is_valid()])