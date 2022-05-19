import argparse
from pathlib import Path

import fastmri_intervention.prepare_data

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

prepare_data = subparsers.add_parser('prepare_data')
prepare_data.add_argument("-i", "--input", type=Path, required=True, #  //umcsanfsclp01.umcn.nl/radng_diag_prostate/archives/Prostate-mpMRI-ScientificArchive/RUMC-2014-20/mri_2014_2020
                 help="Root directory for input, e.g. /path/to/archive/")
prepare_data.add_argument("-o", "--output", type=Path, required=True,
                 help="Root directory for output")
prepare_data.set_defaults(func=lambda args: fastmri_intervention.prepare_data.dcm2mha(Path(args.input), Path(args.output)))

upload_data = subparsers.add_parser('upload_data')
upload_data.add_argument('--api', type=str, required=True)
upload_data.add_argument('--input', type=str, required=True)
upload_data.add_argument('--slug', type=str, required=True)