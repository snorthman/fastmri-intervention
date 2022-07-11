import pickle
from pathlib import Path

import torch, nnunet.inference.predict as predict

from intervention.utils import initialize, settings_schema


def diagnose_fold(fold_dir: Path):
    """
    For each fold, we provide a mean accuracy, intersection-over-union (score), ... https://www.sciencedirect.com/science/article/pii/S1746809420304912
    :param fold_dir:
    :return:
    """
    model = fold_dir / 'model_best.model'
    model_pkl = fold_dir / 'model_best.model.pkl'

    with open(model_pkl, 'rb') as f:
        p = pickle.load(f)
    with open(model, 'rb') as f:
        m = torch.load(f, map_location=torch.device('cpu'))
    #network = nnunet.SegmentationNetwork.load_state_dict()
    predict.predict_cases()


def diagnose(**kwargs):
    archive_dir, dm, gc = initialize('prep', settings_schema, kwargs)

    results_dir = Path(kwargs['results_dir'])

    for item in results_dir.iterdir():
        if 'fold' in item.name and item.is_dir():
            diagnose_fold(item)
