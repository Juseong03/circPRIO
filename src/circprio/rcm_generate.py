"""Sequence-to-RCM generator: the CircCNNs reverse-complementary-match (RCM) descriptor computed from the genome.

The descriptor is that of CircCNNs (Auxiliary_Codes/RCM_features_generation.py, src/third_party/comparator_sources_locked_v1):
exact k-mer reverse-complementary matches for k in KMERS between the last w nt of the upstream flank and the first w nt of the
downstream flank (across), and within each flank, for every w in WINDOWS, counted in a 5 x 5 grid of positional bins and
log1p-transformed -> (3, 5, 250) float32 (branches: across, within upstream, within downstream). The integer counting is the
exact port used in the legacy study (paper_v3_positive_model/fm_research_v1/native_rcm_fast.py, bounded_rcm_features.py),
which was checked there against the CircCNNs code.

Flanks are cut from the genome in transcript orientation (same convention as circprio.windows):
  + strand: upstream = G[start0-3000, start0),        downstream = G[end0, end0+3000)
  - strand: upstream = rc(G[end0, end0+3000)),        downstream = rc(G[start0-3000, start0))
Fixed-flank data sets (CircCNNs-Bench, circ2LO-Bench) use the full 3,000 nt. Intron-bounded data sets (K562-BSJ, GM12878 and
HepG2) cap each flank at the length of the adjacent annotated intron (upstream/downstream intron in transcript orientation, at most
3,000 nt): only the last n nt upstream and the first m nt downstream are counted.
"""
import numpy as np
from numba import njit

from .windows import ALPHABET, rc

FLANK = 3000
WINDOWS = (100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000)
KMERS = (5, 7, 9, 11, 13)
BOUNDED = {'circcnns': False, 'circ2lo': False, 'k562': True, 'GM12878': True, 'HepG2': True}


@njit(cache=True)
def _counts(a, b, k, within):
    n, m = len(a), len(b)
    # (word, position) codes allow counting matches without enumerating repeats.
    bins = np.empty((5, m), dtype=np.int64)
    sizes = np.zeros(5, dtype=np.int64)
    for j in range(m - k + 1):
        word = 0
        for t in range(k):
            word = word * 4 + int(b[j + t])
        col = min(4, ((2 * j + k) * 5) // (2 * m))
        bins[col, sizes[col]] = word * (m + 1) + j
        sizes[col] += 1
    for col in range(5):
        bins[col, :sizes[col]] = np.sort(bins[col, :sizes[col]])
    result = np.zeros((5, 5), dtype=np.int64)
    for i in range(n - k + 1):
        word = 0
        for t in range(k):
            word = word * 4 + 3 - int(a[i + k - 1 - t])
        row = min(4, ((2 * i + k) * 5) // (2 * n))
        lower = word * (m + 1) + (i + k if within else 0)
        upper = (word + 1) * (m + 1)
        for col in range(5):
            values = bins[col, :sizes[col]]
            result[row, col] += np.searchsorted(values, upper) - np.searchsorted(values, lower)
    return result


@njit(cache=True)
def _tensor(pair, up_length, down_length, windows, kmers):
    out = np.empty((3, 5, 250), dtype=np.float32)
    for w in range(len(windows)):
        n, m = min(windows[w], up_length), min(windows[w], down_length)
        a, b = pair[0, FLANK - n:], pair[1, :m]
        for t in range(len(kmers)):
            k = kmers[t]
            parts = (_counts(a, b, k, False), _counts(a, a, k, True), _counts(b, b, k, True))
            for branch in range(3):
                for row in range(5):
                    for col in range(5):
                        out[branch, row, (w * len(kmers) + t) * 5 + col] = np.log1p(np.float32(parts[branch][row, col]))
    return out


def descriptor(pair, up_length=FLANK, down_length=FLANK):
    """RCM (3, 5, 250) float32 of ACGT codes `pair` (2, 3000): [0] upstream flank, [1] downstream flank (transcript orientation)."""
    pair = np.ascontiguousarray(pair, dtype=np.uint8)
    up, down = min(int(up_length), FLANK), min(int(down_length), FLANK)
    if pair.shape != (2, FLANK) or min(up, down) < 1:
        raise ValueError('RCM requires a (2, 3000) flank pair and positive flank lengths')
    if pair[0, FLANK - up:].max() > 3 or pair[1, :down].max() > 3:
        raise ValueError('RCM requires ACGT within the counted flank')
    return _tensor(pair, up, down, np.asarray(WINDOWS), np.asarray(KMERS))


def flanks(fetch, chrom, start0, end0, strand):
    """ACGT codes (2, 3000) in transcript orientation; non-ACGT bases are coded 255. fetch(chrom, a, b) -> genome [a, b)."""
    left = fetch(chrom, start0 - FLANK, start0).upper()
    right = fetch(chrom, end0, end0 + FLANK).upper()
    up, down = (left, right) if strand == '+' else (rc(right), rc(left))
    if len(up) != FLANK or len(down) != FLANK:
        raise ValueError(f'{chrom}:{start0}-{end0} flank runs past the chromosome end')
    return np.array([[ALPHABET.get(b, 255) for b in up], [ALPHABET.get(b, 255) for b in down]], dtype=np.uint8)
