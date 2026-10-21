"""Assemble stored staged-RCM branch/head weights into one inference model."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from circprio.models import CircPrioRCM
from circprio.models.circprio_rcm import Branch, JunctionTrunk, TriRCM, FusionHead


def export(junction, rcm, head, result, output):
    details = json.loads(Path(result).read_text())
    weight = float(details['outer']['class_weight'])
    if weight <= 0:
        raise ValueError('Invalid outer training class weight')
    j, r, h = Branch(JunctionTrunk('meanmax')), Branch(TriRCM()), FusionHead()
    for model, source in [(j, junction), (r, rcm), (h, head)]:
        model.load_state_dict(torch.load(source, map_location='cpu', weights_only=True), strict=True)
    model = CircPrioRCM(pool='meanmax')
    model.junction.load_state_dict(j.trunk.state_dict())
    model.rcm.load_state_dict(r.trunk.state_dict())
    model.head.load_state_dict(h.head.state_dict())
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be empty')
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / 'model.pt')
    (out / 'config.json').write_text(json.dumps({'model': 'rcm', 'pool': 'meanmax', 'length': 400, 'mode': 'joint',
                                                'class_weight': weight}, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--junction', type=Path, required=True, help='junction_meanmax/<seed>/foldN/outer_model.pt')
    p.add_argument('--rcm', type=Path, required=True, help='rcm_branch/<seed>/foldN/outer_model.pt')
    p.add_argument('--head', type=Path, required=True, help='staged_concat_meanmax/<seed>/foldN/outer_head.pt')
    p.add_argument('--result', type=Path, required=True, help='Fusion head result.json from the same dataset/seed/fold')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    export(a.junction, a.rcm, a.head, a.result, a.output)
