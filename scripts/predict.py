"""Predict from prepared per-candidate SpliceBERT states and optional RCM tensors."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from circprio.models import CircPrio, CircPrioRCM


def predict(inputs, checkpoint, config, output, device='cpu', batch_size=64):
    cfg = json.loads(Path(config).read_text())
    if cfg.get('model') not in ('junction', 'rcm') or cfg.get('pool') != 'meanmax':
        raise ValueError('Expected model junction/rcm with pool meanmax')
    weight = float(cfg['class_weight'])
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError('class_weight must be the positive training class weight')
    model = CircPrio(pool='meanmax') if cfg['model'] == 'junction' else CircPrioRCM(pool='meanmax')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model = model.eval().to(device)
    with np.load(inputs, allow_pickle=False) as arrays:
        a, d = arrays['acceptor'], arrays['donor']
        if a.shape != d.shape or a.ndim != 3 or a.shape[1:] != (400, 512) or not len(a):
            raise ValueError('acceptor/donor must each be nonempty (N, 400, 512) arrays')
        r = arrays['rcm'] if cfg['model'] == 'rcm' else None
        if r is not None and r.shape != (len(a), 3, 5, 250):
            raise ValueError('rcm must have shape (N, 3, 5, 250) in the same candidate order')
        if not all(np.isfinite(x).all() for x in (a, d) + (() if r is None else (r,))):
            raise ValueError('Input features contain nonfinite values')
        ids = arrays['candidate_id'] if 'candidate_id' in arrays else np.arange(len(a))
        if ids.ndim != 1 or len(ids) != len(a):
            raise ValueError('candidate_id must contain N entries')
        values = []
        with torch.inference_mode():
            for start in range(0, len(a), batch_size):
                tensor = lambda x: torch.from_numpy(np.array(x[start:start + batch_size], dtype=np.float32, copy=True)).to(device)
                kwargs = dict(acceptor=tensor(a), donor=tensor(d))
                if r is not None:
                    kwargs['rcm_raw'] = tensor(r)
                logits = model(**kwargs)
                z = logits[:, 1] - logits[:, 0]
                values.append(torch.stack((z, torch.sigmoid(z - np.log(weight))), 1).cpu().numpy())
    values = np.concatenate(values)
    out = pd.DataFrame({'candidate_id': ids, 'logit': values[:, 0], 'score': values[:, 1], 'threshold': 1 / (1 + weight)})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('outputs/predictions.csv'))
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=64)
    a = p.parse_args()
    if a.batch_size < 1:
        p.error('--batch-size must be positive')
    predict(a.inputs, a.checkpoint, a.config, a.output, a.device, a.batch_size)
