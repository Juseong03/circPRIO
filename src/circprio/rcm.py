"""Inputs of circPRIO-RCM and its RCM-only controls: the CircCNNs reverse-complementary-match (RCM) descriptor per candidate.

RCM arrays (N, 3, 5, 250) float32, log-transformed counts in candidate order (legacy seqjcf.dataset):
  circcnns  native CircCNNs RCM of the fixed 3-kb benchmark flanks (legacy fm_legacy_native_inputs_v1_20260910; row order checked
            against the candidate table through rcm_indices.npy)
  circ2lo   fixed 3-kb windows outside each end (circ2LO-Bench/rcm.npy)
  k562      intron-bounded flanks capped at 3 kb (K562-flank-RCM-features/bounded_rcm.npy)
  GM12878, HepG2  intron-bounded flanks capped at 3 kb, built with the K562 workflow (GM12878_HepG2-Disjoint/<cohort>/rcm.npy)
`rcm_raw` is the array as stored (TriRCM, CircCNNs RCM model); `rcm` is standardised with the mean and SD of the training rows of
the current phase (RCM-only MLP; legacy seqjcf.Bank). Branch features are the 128-d outputs of trained, frozen branch trunks.
"""
import numpy as np
import pandas as pd
import torch

from . import DATA, data

SOURCES = {name: DATA / 'datasets' / name / 'rcm.npy' for name in data.DATASETS}


def load(name):
    if not SOURCES[name].exists():
        raise FileNotFoundError(f'{SOURCES[name]}: download the RCM-only ZIP from the Drive link in data/README.md and extract into the repository root')
    rcm = np.load(SOURCES[name], mmap_mode='r')
    expected = (len(data.frame(name)), 3, 5, 250)
    if rcm.shape != expected:
        raise ValueError(f'{name}: RCM shape {rcm.shape} != {expected}')
    return np.ascontiguousarray(rcm)


class RCMInputs:
    """Batches with `rcm_raw`, `rcm` (phase-standardised) and, given `junction` (training.JunctionInputs), acceptor/donor states."""

    def __init__(self, name, device, junction=None):
        self.raw, self.device, self.junction = load(name), device, junction
        self.y = data.frame(name).y.to_numpy()
        self.mean = self.scale = None

    def use(self, parts, phase):
        rows = np.sort(parts['inner_train' if phase == 'inner' else 'outer_train'])
        flat = self.raw[rows].reshape(len(rows), -1)
        sd = flat.std(0)
        self.mean, self.scale = flat.mean(0).astype(np.float32), np.where(sd < 1e-8, 1., sd).astype(np.float32)

    def batch(self, rows):
        rows = np.sort(rows)
        raw = self.raw[rows]
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(self.device, non_blocking=True)
        out = dict(rcm_raw=t(raw), rcm=t(((raw.reshape(len(rows), -1) - self.mean) / self.scale).reshape(raw.shape)))
        if self.junction is not None:
            out.update(self.junction.batch(rows)[1])
        return rows, out


class FeatureInputs:
    """Frozen branch features for training a fusion head: features[phase] (N, d) filled for the rows that phase may use."""

    def __init__(self, features, y, device):
        self.features, self.y, self.device, self.current = features, y, device, None

    def use(self, parts, phase):
        self.current = self.features[phase]

    def batch(self, rows):
        rows = np.sort(rows)
        return rows, dict(features=torch.from_numpy(np.ascontiguousarray(self.current[rows])).to(self.device))


def branch_features(trunk, inputs, rows, batch=512):
    """128-d trunk outputs (eval mode) for `rows`, in the order of `rows` sorted."""
    trunk.eval()
    order, out = np.sort(rows), []
    with torch.inference_mode():
        for s in range(0, len(order), batch):
            _, x = inputs.batch(order[s:s + batch])
            out.append(trunk(**x).float().cpu().numpy())
    return order, np.concatenate(out)
