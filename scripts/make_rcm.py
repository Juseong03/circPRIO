"""Compute the CircCNNs RCM descriptor of a dataset from a reference genome (src/circprio/rcm_generate.py).

Genomes are not distributed: circcnns and circ2lo use hg19, k562, GM12878 and HepG2 use hg38 (indexed FASTA, pyfaidx).
K562, GM12878 and HepG2 cap each flank at the adjacent annotated intron (data/datasets/<dataset>/rcm_flanks.parquet).

  python scripts/make_rcm.py --dataset k562 --genome /path/hg38.fa [--out data/datasets/k562/rcm.npy] [--workers 16]
  python scripts/make_rcm.py --dataset k562 --genome /path/hg38.fa --check data/datasets/k562/rcm.npy   (compare instead of write)
"""
import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from circprio import DATA, data  # noqa: E402
from circprio import rcm_generate as G  # noqa: E402
from circprio.windows import Genome  # noqa: E402

_fa = None


def _init(genome):
    global _fa
    _fa = Genome(genome)._fa


def _job(item):
    chrom, start0, end0, strand, up, down = item
    return G.descriptor(G.flanks(lambda c, a, b: str(_fa[c][a:b]), chrom, start0, end0, strand), up, down)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=data.DATASETS, required=True)
    p.add_argument('--genome', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--check', type=Path, help='existing rcm.npy to compare with (nothing is written)')
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    f = data.frame(a.dataset)
    if G.BOUNDED[a.dataset]:
        lengths = pd.read_parquet(DATA / 'datasets' / a.dataset / 'rcm_flanks.parquet')
        if not np.array_equal(lengths.candidate_id.to_numpy(), f.candidate_id.to_numpy()):
            raise ValueError('rcm_flanks.parquet is not in candidate order')
        up, down = lengths.upstream_intron_nt.to_numpy(), lengths.downstream_intron_nt.to_numpy()
    else:
        up = down = np.full(len(f), G.FLANK)
    items = [(r.chromosome, int(r.start0), int(r.end0), r.strand, int(u), int(d)) for r, u, d in zip(f.itertuples(index=False), up, down)]
    G.descriptor(np.zeros((2, G.FLANK), np.uint8))           # compile once before forking
    with mp.get_context('fork').Pool(a.workers, _init, (a.genome,)) as pool:
        rcm = np.stack(pool.map(_job, items, chunksize=32))
    if a.check:
        stored = np.load(a.check, mmap_mode='r')
        diff = np.abs(rcm - stored).reshape(len(rcm), -1).max(1)
        print(f'{a.dataset}: {int((diff == 0).sum())}/{len(rcm)} candidates identical, max |difference| {diff.max():g}')
    else:
        out = a.out or DATA / 'datasets' / a.dataset / 'rcm.npy'
        np.save(out, rcm)
        print(f'{a.dataset}: wrote {out} {rcm.shape}')


if __name__ == '__main__':
    main()
