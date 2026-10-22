"""RCM generator: counting rules on small sequences, flank orientation, and agreement with the distributed arrays when present."""
import numpy as np
import pandas as pd
import pytest

from circprio import DATA, data, rcm_generate as G
from circprio.windows import ALPHABET, rc


def codes(seq):
    return np.array([ALPHABET[b] for b in seq], dtype=np.uint8)


def test_across_match_is_counted_once_and_within_needs_no_overlap():
    rng = np.random.default_rng(0)
    pair = rng.integers(0, 4, size=(2, G.FLANK)).astype(np.uint8)
    base = G.descriptor(pair)
    word = 'ACGTTGCAAGT'                                  # 11-mer placed at the 3' end upstream, its reverse complement downstream
    pair[0, -11:], pair[1, :11] = codes(word), codes(rc(word))
    out = G.descriptor(pair)
    k11 = G.KMERS.index(11)
    pos = (0 * len(G.KMERS) + k11) * 5 + 0                 # window 100 nt, 11-mer, first downstream positional bin
    assert np.expm1(out[0, 4, pos]) >= np.expm1(base[0, 4, pos]) + 1
    assert out.shape == (3, 5, 250) and out.dtype == np.float32


def test_flank_orientation():
    genome = {'c': 'A' * 3000 + 'C' * 100 + 'G' * 3000}
    fetch = lambda c, a, b: genome[c][a:b]
    plus = G.flanks(fetch, 'c', 3000, 3100, '+')
    minus = G.flanks(fetch, 'c', 3000, 3100, '-')
    assert (plus[0] == ALPHABET['A']).all() and (plus[1] == ALPHABET['G']).all()
    assert (minus[0] == ALPHABET['C']).all() and (minus[1] == ALPHABET['T']).all()


def test_bounded_flank_ignores_sequence_beyond_the_intron():
    rng = np.random.default_rng(1)
    pair = rng.integers(0, 4, size=(2, G.FLANK)).astype(np.uint8)
    other = pair.copy()
    other[0, :-700], other[1, 900:] = 255, 255             # outside the counted flanks
    assert np.array_equal(G.descriptor(pair, 700, 900), G.descriptor(other, 700, 900))


@pytest.mark.parametrize('name', [n for n in data.DATASETS if G.BOUNDED[n]])
def test_flank_tables_match_candidates(name):
    lengths = pd.read_parquet(DATA / 'datasets' / name / 'rcm_flanks.parquet')
    assert np.array_equal(lengths.candidate_id.to_numpy(), data.frame(name).candidate_id.to_numpy())
    assert (lengths[['upstream_intron_nt', 'downstream_intron_nt']] > 0).all().all()
