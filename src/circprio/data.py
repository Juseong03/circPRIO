"""Portable prepared inputs; all paths resolve within the checkout."""
import json
from functools import lru_cache
import numpy as np
import pandas as pd
from . import ARTIFACTS, DATA

DATASETS = ('circcnns', 'circ2lo', 'k562', 'GM12878', 'HepG2')
WINDOW_CACHE = DATA / 'windows'


@lru_cache(maxsize=None)
def frame(name):
    if name not in DATASETS:
        raise KeyError(name)
    path = DATA / 'datasets' / name / 'candidates.parquet'
    if not path.exists():
        raise FileNotFoundError(f'{path}: candidate table is missing from the checkout (data/README.md)')
    f = pd.read_parquet(path)
    columns = ['candidate_id', 'chromosome', 'start0', 'end0', 'strand', 'y', 'component', 'fold']
    if not set(columns).issubset(f.columns) or not f.candidate_id.is_unique:
        raise ValueError('Candidate table must have unique IDs and the documented columns')
    if f[columns].isna().any().any() or not set(f.y.unique()) <= {0, 1}:
        raise ValueError('Candidate table contains missing values or invalid labels')
    if name in DATASETS[:3]:
        if set(f.fold.unique()) != set(range(5)) or f.groupby('component').fold.nunique().max() != 1:
            raise ValueError('Expected five component-disjoint fixed folds')
    return f.reset_index(drop=True)


def partitions(name, fold):
    if name not in DATASETS[:3] or fold not in range(5):
        raise ValueError('Training requires a training dataset and fold 0..4')
    folds = frame(name).fold.to_numpy()
    sel = (fold + 1) % 5
    return dict(test=np.flatnonzero(folds == fold), selection=np.flatnonzero(folds == sel),
                inner_train=np.flatnonzero((folds != fold) & (folds != sel)), outer_train=np.flatnonzero(folds != fold))


def windows(name, length):
    path = WINDOW_CACHE / f'{name}_L{length}.npz'
    meta = path.with_suffix('.json')
    if not path.exists() or not meta.exists():
        raise FileNotFoundError(f'{path}: compressed windows or their JSON manifest are missing (data/README.md)')
    manifest = json.loads(meta.read_text())
    with np.load(path, allow_pickle=False) as archive:
        array = archive['windows']
    if (manifest['dataset'], manifest['length'], manifest['rows']) != (name, length, len(frame(name))):
        raise ValueError('Window manifest does not match candidate table')
    if array.shape != (len(frame(name)), 2, length) or array.dtype != np.uint8:
        raise ValueError('Expected uint8 windows with shape (N, 2, length)')
    return array
