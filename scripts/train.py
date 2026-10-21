"""Fit circPRIO-J or staged circPRIO-RCM for one fixed fold/seed; export a complete inference checkpoint."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from circprio import data
from circprio.training import JunctionInputs, train_phase, logits
from circprio.rcm import RCMInputs, FeatureInputs, branch_features
from circprio.models import CircPrioJ, CircPrioRCM
from circprio.models.circprio_rcm import Branch, JunctionTrunk, TriRCM, FusionHead


def train(model, dataset, fold, seed, output, device):
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be empty; use a new fold/seed directory')
    out.mkdir(parents=True, exist_ok=True)
    parts, f = data.partitions(dataset, fold), data.frame(dataset)
    junction = JunctionInputs(dataset, 400, 'joint', device)

    def refit(make, inputs):
        inner_net, inner = train_phase(make, inputs, parts, seed, 'inner', device=device)
        outer_net, outer = train_phase(make, inputs, parts, seed, 'outer', inner['selected_epoch'], device=device)
        return inner_net, outer_net, inner, outer

    details = {}
    if model == 'junction':
        _, net, inner, outer = refit(lambda: CircPrioJ(pool='meanmax'), junction)
        inputs = junction
    else:
        rcm = RCMInputs(dataset, device)
        features = {phase: [] for phase in ('inner', 'outer')}
        trunks = []
        for name, make, inputs in [('junction', lambda: Branch(JunctionTrunk('meanmax')), junction),
                                   ('rcm', lambda: Branch(TriRCM()), rcm)]:
            im, om, bi, bo = refit(make, inputs)
            details[name] = {'inner': bi, 'outer': bo}
            for phase, m, rows in [('inner', im, np.concatenate([parts['inner_train'], parts['selection']])),
                                   ('outer', om, np.concatenate([parts['outer_train'], parts['test']]))]:
                ix, values = branch_features(m.trunk, inputs, rows)
                full = np.zeros((len(f), 128), dtype=np.float32)
                full[ix] = values
                features[phase].append(full)
            trunks.append(om.trunk)
        inputs = FeatureInputs({k: np.concatenate(v, axis=1) for k, v in features.items()}, f.y.to_numpy(), device)
        _, head, inner, outer = refit(FusionHead, inputs)
        net = CircPrioRCM(pool='meanmax').to(device)
        net.junction.load_state_dict(trunks[0].state_dict())
        net.rcm.load_state_dict(trunks[1].state_dict())
        net.head.load_state_dict(head.head.state_dict())
        inputs = RCMInputs(dataset, device, junction=junction)
    inputs.use(parts, 'outer') if hasattr(inputs, 'use') else None
    ix, z = logits(net, inputs, parts['test'])
    weight = outer['class_weight']
    torch.save(net.state_dict(), out / 'model.pt')
    (out / 'config.json').write_text(json.dumps({'model': model, 'pool': 'meanmax', 'length': 400, 'mode': 'joint',
                                                'class_weight': weight, 'dataset': dataset, 'fold': fold, 'seed': seed}, indent=2)+'\n')
    (out / 'result.json').write_text(json.dumps({'inner': inner, 'outer': outer, 'branches': details}, indent=2)+'\n')
    score = torch.sigmoid(torch.as_tensor(z) - np.log(weight)).numpy()
    pd.DataFrame({'candidate_id': f.candidate_id.to_numpy()[ix], 'y': f.y.to_numpy()[ix], 'logit': z,
                  'score': score, 'threshold': 1 / (1 + weight)}).to_csv(out / 'test_predictions.csv', index=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['junction', 'rcm'], required=True)
    p.add_argument('--dataset', choices=['circcnns', 'circ2lo', 'k562'], required=True)
    p.add_argument('--fold', type=int, choices=range(5), default=0)
    p.add_argument('--seed', type=int, default=20260910)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    train(a.model, a.dataset, a.fold, a.seed, a.output, a.device)
