"""Generate 400-nt joint embeddings from externally distributed window arrays."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from circprio.embedding import embed
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataset', choices=['circcnns', 'circ2lo', 'k562', 'GM12878', 'HepG2'], required=True)
p.add_argument('--device', default='cuda:0')
a = p.parse_args()
print(embed(a.dataset, 400, 'joint', device=a.device))
