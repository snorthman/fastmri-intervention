import argparse, json
from pathlib import Path

from prep.workflow import workflow


def run_prepare(args):
    with open(args.json) as j:
        settings = json.load(j)
    workflow(Path(args.pelvis), Path(args.archive) if args.archive else None, **settings)


parser = argparse.ArgumentParser()
parser.add_argument("--json", type=Path, required=True,
                 help="Path to JSON settings file")
parser.add_argument("--pelvis", type=Path, required=True,
                 help="chamsey/pelvis root directory")
parser.add_argument("--archive", type=Path, required=False,
                 help="radng_diag_prostate root directory")
parser.set_defaults(func=run_prepare)


if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)