"""Splice-site windows in transcript orientation.

Convention (verified against all three legacy data sets, Milestone 1 audit): windows are cut from the *genome*, centred on the
back-splice coordinates, not from the spliced transcript. For a candidate spanning genomic [start0, end0) on strand s and a
per-site window of length L (half = L/2):

  + strand: acceptor = G[start0-half, start0+half),            donor = G[end0-half, end0+half)
  - strand: acceptor = rc(G[end0-half, end0+half)),            donor = rc(G[start0-half, start0+half))

so the acceptor window is half upstream intron + half from the start of the circularised span, and the donor window is half
from the end of the span + half downstream intron (5'->3' transcript orientation). If the first or last exon of the span is
shorter than `half`, the 'exonic' half continues into the following intron (genomic, not spliced sequence); if an intron is
shorter than `half`, the 'intronic' half reaches into the neighbouring exon; if the span is shorter than 2*half the two windows
overlap. `boundary_flags` reports these cases; the population is never changed silently.
"""
import gzip
from pathlib import Path

import numpy as np

_RC = str.maketrans('ACGTNacgtn', 'TGCANtgcan')
ALPHABET = {b: i for i, b in enumerate('ACGT')}


def rc(seq):
    return seq.translate(_RC)[::-1]


def site_windows(fetch, chrom, start0, end0, strand, length):
    """fetch(chrom, a, b) -> genomic sequence [a, b) (upper case). Returns (acceptor, donor) in transcript orientation."""
    if length % 2:
        raise ValueError('window length must be even')
    half = length // 2
    left = fetch(chrom, start0 - half, start0 + half).upper()
    right = fetch(chrom, end0 - half, end0 + half).upper()
    return (left, right) if strand == '+' else (rc(right), rc(left))


def encode(seq):
    return np.fromiter((ALPHABET[b] for b in seq), dtype=np.uint8, count=len(seq))


def valid(seq, length):
    return len(seq) == length and set(seq) <= set('ACGT')


class Genome:
    """Random access to a reference: indexed FASTA (pyfaidx) or a folder of per-chromosome .fa.gz files (loaded lazily)."""

    def __init__(self, path):
        self.path = Path(path)
        self._fa, self._chroms = None, {}
        if self.path.is_file():
            import pyfaidx
            self._fa = pyfaidx.Fasta(str(self.path), as_raw=True, sequence_always_upper=True)

    def length(self, chrom):
        return len(self._fa[chrom]) if self._fa is not None else len(self._load(chrom))

    def _load(self, chrom):
        if chrom not in self._chroms:
            with gzip.open(self.path / f'{chrom}.fa.gz', 'rt') as handle:
                self._chroms[chrom] = ''.join(line.strip() for line in handle if not line.startswith('>')).upper()
        return self._chroms[chrom]

    def fetch(self, chrom, a, b):
        if a < 0:
            raise IndexError('window starts before the chromosome start')
        seq = self._fa[chrom][a:b] if self._fa is not None else self._load(chrom)[a:b]
        if len(seq) != b - a:
            raise IndexError('window runs past the chromosome end')
        return seq


def boundary_flags(span, first_exon, last_exon, up_intron, down_intron, length):
    """Boolean flags for one candidate at per-site window `length` (NaN-safe for unknown exon/intron lengths)."""
    half = length // 2
    out = dict(windows_overlap=span < 2 * half)
    for name, value in (('first_exon_shorter_than_half', first_exon), ('last_exon_shorter_than_half', last_exon),
                        ('upstream_intron_shorter_than_half', up_intron), ('downstream_intron_shorter_than_half', down_intron)):
        out[name] = bool(value < half) if value == value else None
    return out
