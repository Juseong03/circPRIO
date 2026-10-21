"""Prepared, committed data match original raw-window hashes and fixed folds."""
import hashlib
import json
import numpy as np
import pytest
from circprio import data, rcm


@pytest.mark.parametrize('name', data.DATASETS)
def test_committed_candidates_and_windows(name):
    f = data.frame(name)
    w = data.windows(name, 400)
    meta = json.loads((data.WINDOW_CACHE / f'{name}_L400.json').read_text())
    assert w.shape == (len(f), 2, 400)
    assert w.dtype == np.uint8
    assert set(np.unique(w)) <= {0, 1, 2, 3, 255}
    assert hashlib.sha256(w.tobytes()).hexdigest() == meta['sha256']
    if name in data.DATASETS[:3]:
        for fold in range(5):
            parts = data.partitions(name, fold)
            groups = [set(parts[k]) for k in ('inner_train', 'selection', 'test')]
            assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            assert len(set.union(*groups)) == len(f)
    else:
        assert set(f.fold.unique()) == {-1}


def test_external_rcm_has_clear_download_instructions(tmp_path, monkeypatch):
    monkeypatch.setitem(rcm.SOURCES, 'k562', tmp_path / 'rcm.npy')
    with pytest.raises(FileNotFoundError, match='RCM-only ZIP'):
        rcm.load('k562')
