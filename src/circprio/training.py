"""Inner epoch selection and outer refit utilities for circPRIO.

For outer fold f: inner training on the three folds other than f and (f+1) mod 5, epoch selection by AP on fold (f+1) mod 5
(at most 20 epochs, at least 3, patience 5, min_delta 1e-6; the best inner state is kept), then a fresh model is trained on all
four outer-training folds for the selected number of epochs and scores fold f. AdamW (lr 3e-4, weight decay 1e-4), batch 64,
gradient clipping 5, class-weighted cross-entropy (weight n0/n1 on positives), seed-controlled permutation order.
Reported score: prior-corrected probability sigmoid(z - log w) with z = logit_1 - logit_0; class-balanced boundary 1/(1 + w).
"""
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

from . import data
from .embedding import States

TRAIN = dict(lr=3e-4, weight_decay=1e-4, batch_size=64, max_epochs=20, min_epochs=3, patience=5, gradient_clip=5.0, min_delta=1e-6)


class JunctionInputs:
    """Batches of cached SpliceBERT states (acceptor, donor) for one data set / window length / encoding mode."""

    def __init__(self, name, length, mode, device):
        self.states, self.device = States(name, length, mode), device
        self.y = data.frame(name).y.to_numpy()

    def batch(self, rows):
        rows = np.sort(rows)
        a, d = self.states.get(rows)
        # fp16 is copied to the GPU and widened there (exact conversion, identical values; less CPU work than widening on the host)
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(self.device, non_blocking=True).float()
        return rows, dict(acceptor=t(a), donor=t(d))


def logits(net, inputs, rows, batch=512):
    net.eval()
    order, out = np.sort(rows), np.empty(len(rows))
    with torch.inference_mode():
        for s in range(0, len(order), batch):
            ix, x = inputs.batch(order[s:s + batch])
            z = net(**x)
            out[s:s + len(ix)] = (z[:, 1] - z[:, 0]).float().cpu().numpy()
    return order, out


def batches(rows, size):
    """Consecutive batches; a trailing single row joins the previous batch (BatchNorm needs two rows; otherwise unchanged)."""
    parts = [rows[i:i + size] for i in range(0, len(rows), size)]
    if len(parts) > 1 and len(parts[-1]) == 1:
        parts[-2] = np.concatenate([parts[-2], parts.pop()])
    return parts


def train_phase(make_net, inputs, parts, seed, phase, epochs=None, device='cuda:0', net=None, settings=None):
    """net: a model built before seeding (legacy initialisation order); otherwise make_net() is called after seeding.
    settings: training settings overriding TRAIN (control models with their own published settings; optimizer 'AdamW' or 'Adam').
    Inputs with a `use(parts, phase)` method (RCM standardisation, cached branch features) are switched to the phase first."""
    cfg = {**TRAIN, **(settings or {})}
    if hasattr(inputs, 'use'):
        inputs.use(parts, phase)
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = (net if net is not None else make_net()).to(device)
    Optimizer = torch.optim.Adam if cfg.get('optimizer', 'AdamW') == 'Adam' else torch.optim.AdamW
    opt = Optimizer([p for p in net.parameters() if p.requires_grad], lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    rows = parts['inner_train' if phase == 'inner' else 'outer_train']
    y_all = inputs.y
    weight = float((y_all[rows] == 0).sum() / (y_all[rows] == 1).sum())
    rng = np.random.default_rng(seed)
    best, chosen, state, history = -np.inf, 0, None, []
    started = time.monotonic()
    for epoch in range(1, (cfg['max_epochs'] if phase == 'inner' else epochs) + 1):
        net.train()
        order = rng.permutation(rows)
        for b in batches(order, cfg['batch_size']):
            ix, x = inputs.batch(b)
            y = torch.as_tensor(y_all[ix], device=device)
            loss = F.cross_entropy(net(**x), y, reduction='none')
            loss = (loss * torch.where(y == 1, weight, 1.)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['gradient_clip'])
            opt.step()
        record = dict(epoch=epoch, seconds=time.monotonic() - started)
        if phase == 'inner':
            sel, z = logits(net, inputs, parts['selection'])
            record['selection_AP'] = float(average_precision_score(y_all[sel], z))
            if record['selection_AP'] > best + cfg['min_delta']:
                best, chosen = record['selection_AP'], epoch
                state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            if epoch >= cfg['min_epochs'] and epoch - chosen >= cfg['patience']:
                history.append(record)
                break
        history.append(record)
    if phase == 'inner':
        net.load_state_dict(state)
    return net, dict(selected_epoch=chosen if phase == 'inner' else epochs, selection_AP=best if phase == 'inner' else None,
                     class_weight=weight, history=history, seconds=time.monotonic() - started)

