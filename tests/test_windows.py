"""Window extraction: orientation, lengths, centring and reverse complement (Milestone 1)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from circprio.windows import rc, site_windows, boundary_flags  # noqa: E402

GENOME = 'AAAACCCCGGGGTTTTACGTACGTAAAACCCCGGGGTTTT'          # 40 nt toy chromosome


def fetch(chrom, a, b):
    return GENOME[a:b]


def test_plus_strand_windows_are_centred_on_the_coordinates():
    acceptor, donor = site_windows(fetch, 'c', 10, 30, '+', 8)
    assert acceptor == GENOME[6:14] and donor == GENOME[26:34]


def test_minus_strand_swaps_and_reverse_complements():
    acceptor, donor = site_windows(fetch, 'c', 10, 30, '-', 8)
    assert acceptor == rc(GENOME[26:34]) and donor == rc(GENOME[6:14])


@pytest.mark.parametrize('length', [2, 4, 8, 12])
def test_window_lengths(length):
    for strand in '+-':
        a, d = site_windows(fetch, 'c', 15, 25, strand, length)
        assert len(a) == len(d) == length


def test_odd_length_rejected():
    with pytest.raises(ValueError):
        site_windows(fetch, 'c', 10, 30, '+', 7)


def test_rc_is_an_involution_and_not_a_reversal():
    s = 'ACGTTGCA'
    assert rc(rc(s)) == s and rc('AACG') == 'CGTT' and rc('AACG') != 'AACG'[::-1]


def test_boundary_flags():
    f = boundary_flags(span=150, first_exon=40, last_exon=120, up_intron=500, down_intron=float('nan'), length=200)
    assert f['windows_overlap'] and f['first_exon_shorter_than_half'] and not f['last_exon_shorter_than_half']
    assert f['upstream_intron_shorter_than_half'] is False and f['downstream_intron_shorter_than_half'] is None
