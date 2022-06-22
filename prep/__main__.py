import argparse, json
from pathlib import Path

from prep.workflow import workflow


def cli():
    args = parser.parse_args()
    args.func(args)


def run_prepare(args):
    with open(args.json) as j:
        settings = json.load(j)
    workflow(**settings)


parser = argparse.ArgumentParser()
parser.add_argument("--json", type=Path, required=True,
                 help="Path to workflow.json settings file")
parser.set_defaults(func=run_prepare)