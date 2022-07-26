import logging
from pathlib import Path
from datetime import datetime

import click

import intervention.segment.inference
from intervention.utils import Settings
from intervention.prep.convert import dcm2mha, generate_dcm2mha_json, mha2nnunet
from intervention.prep.upload import upload_data, delete_all_data
from intervention.prep.annotate import write_annotations
from intervention.utils import DirectoryManager, GCAPI, Settings
from intervention.segment.inference import inference, plot


def upload(dm: DirectoryManager, gc: GCAPI):
    if click.confirm('Confirm delete? (required when uploading)'):
        logging.info(f'Deleting mha files @ grand-challenge.org/reader-studies/{gc.slug}')
        delete_all_data(gc)
        logging.info(f'Uploading mha files @ {dm.mha} to grand-challenge.org/reader-studies/{gc.slug}')
        upload_data(dm.mha, gc)
    else:
        logging.info('Cancelled delete, skipping upload step')


def print_actions(actions: list):
    len_out = max([len(out) for _, _, out, _ in actions])
    for i, (name, desc, out, _) in enumerate(actions):
        out = out + ' < ' if out else ''
        print(f'{str(i + 1).rjust(len(str(len(actions))))}:\t{name.ljust(10)}\t{out.ljust(len_out)}{desc}')


@click.command()
@click.option('-s', '--settings', type=click.Path(resolve_path=True, path_type=Path),
              prompt='Enter path/to/settings.json', default='.', help="Path to json settings file")
def cli(settings: Path):
    s = Settings(settings)
    dm, gc, archive_dir = s.dm, s.gc, s.archive_dir
    dcm_json = dm.dcm / "dcm2mha_settings.json"

    s.summary()

    actions = []
    actions.append(('dcm', f'create a DICOM2MHA json settings file using {archive_dir}',
                    f'{dcm_json}',
                    lambda: generate_dcm2mha_json(dm, archive_dir)))
    actions.append(('dcm2mha', f'convert all DICOM in {archive_dir} to MHA using {dcm_json}',
                    f'{dm.mha}/*',
                    lambda: dcm2mha(dm, archive_dir)))
    actions.append(('upload', f'delete all cases in {gc.slug} and upload all MHA in {dm.mha}',
                    '',
                    lambda: upload(dm, gc)))
    actions.append(('annotate', f'download and process all annotations from {gc.slug}',
                    f'{dm.annotations}\\*',
                    lambda: write_annotations(dm, gc)))
    actions.append(('mha2nnunet', f'convert all MHA + annotations in {dm.mha} and {dm.annotations} to nnUNet',
                    f'{dm.nnunet}\\*',
                    lambda: mha2nnunet(dm, s.test_percentage)))
    actions.append(('inference', f'using all DICOM and MHA in {dm.predict} using {dm.task_dirname}\\{s.trainer} in {dm.results}',
                    f'{dm.predict}\\inference_*\\*',
                    lambda: inference(s)))
    actions.append(('plot', f'generate plots in each inference directory ({dm.predict}\\inference_*)',
                    '',
                    lambda: plot(dm)))

    print_actions(actions)

    action_funcs = {name: func for name, _, _, func in actions}

    choices = list(action_funcs.keys()) + ['exit']
    while 1:
        try:
            inp: str = click.prompt(f'\nExecute functions in order: [{", ".join(choices)}] (exit closes the program immediately)', type=str)
            if 'exit' in inp:
                return

            inp_split = inp.split(' ')
            run = []
            for i in inp_split:
                if i in action_funcs.keys():
                    run.append(action_funcs[i])
                else:
                    raise KeyError(f'unknown function: {i}')

            print('\nSelected functions:')
            print_actions([action for action in actions if action[0] in inp_split])
            click.confirm('\nStart program?', abort=True)

            start = datetime.now()
            logging.info(f"Program started at {start}")

            for func in run:
                func()

            end = datetime.now()
            logging.info(f"Program end at {end}\n\truntime {end - start}")
            return
        except (click.exceptions.Abort, KeyError) as e:
            continue




if __name__ == '__main__':
    cli()

# @cli.command(name='prep')
# @click.option('-s', '--settings', type=click.Path(resolve_path=True, path_type=Path),
#               prompt='Enter path/to/settings.json', default='.', help="Path to json settings file")
# def prep(settings: Path):
#     intervention.prep.prep(Settings('prep', settings))
#
#
# @cli.command(name='inference')
# @click.option('-s', '--settings', type=click.Path(resolve_path=True, path_type=Path),
#               prompt='Enter path/to/settings.json', default='.', help="Path to json settings file")
# def inference(settings: Path):
#     inference_dir = intervention.segment.inference(Settings('inference', settings))
#     intervention.segment.plot(inference_dir)
#
#
# @cli.command(name='plot')
# @click.option('-s', '--settings', type=click.Path(resolve_path=True, path_type=Path),
#               prompt='Enter path/to/settings.json', default='.', help="Path to json settings file")
# def plot(settings: Path):
#     s = Settings('plot', settings)
#     logging.info(f'iterating through {s.dm.predict} for prediction directories')
#     for d in s.dm.predict.iterdir():
#         if d.is_dir() and d.name.startswith('inference_'):
#             try:
#                 intervention.segment.plot(d)
#             except Exception as e:
#                 logging.error(str(e))