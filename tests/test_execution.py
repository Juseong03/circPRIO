"""Inference consistency and minimal training on synthetic data; no external assets."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from circprio.models import CircPrio, CircPrioRCM
from circprio.models.circprio_rcm import Branch, JunctionTrunk, TriRCM, FusionHead
from predict import predict
from export_checkpoint import export
import train as training_cli
from circprio import training


@pytest.mark.parametrize('kind', ['junction', 'rcm'])
def test_prediction_matches_direct_model(tmp_path, kind):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    net = (CircPrio(pool='meanmax') if kind == 'junction' else CircPrioRCM(pool='meanmax')).eval()
    a, d = np.random.default_rng(7).normal(size=(2, 2, 400, 512)).astype(np.float32)
    r = np.zeros((2, 3, 5, 250), dtype=np.float32)
    np.savez(tmp_path/'features.npz', acceptor=a, donor=d, rcm=r, candidate_id=np.array(['a', 'b']))
    torch.save(net.state_dict(), tmp_path/'model.pt')
    (tmp_path/'config.json').write_text(json.dumps({'model': kind, 'pool': 'meanmax', 'class_weight': 9}))
    actual = predict(tmp_path/'features.npz', tmp_path/'model.pt', tmp_path/'config.json', tmp_path/'out.csv', batch_size=1)
    kwargs = dict(acceptor=torch.from_numpy(a), donor=torch.from_numpy(d))
    if kind == 'rcm': kwargs['rcm_raw'] = torch.from_numpy(r)
    with torch.inference_mode():
        z = net(**kwargs)
        expected = torch.sigmoid(z[:, 1] - z[:, 0] - np.log(9)).numpy()
    np.testing.assert_allclose(actual.score, expected, atol=1e-6)
    assert actual.candidate_id.tolist() == ['a', 'b']
    np.testing.assert_allclose(actual.threshold, 0.1)


def test_staged_checkpoint_export(tmp_path):
    torch.set_num_threads(2)
    j, r, h = Branch(JunctionTrunk('meanmax')), Branch(TriRCM()), FusionHead()
    for name, model in [('j',j),('r',r),('h',h)]: torch.save(model.state_dict(), tmp_path/f'{name}.pt')
    result = tmp_path/'result.json'; result.write_text(json.dumps({'outer':{'class_weight': 3}}))
    export(tmp_path/'j.pt', tmp_path/'r.pt', tmp_path/'h.pt', result, tmp_path/'export')
    net = CircPrioRCM(pool='meanmax')
    net.load_state_dict(torch.load(tmp_path/'export/model.pt', weights_only=True))
    for x, y in zip(net.junction.parameters(), j.trunk.parameters()): assert torch.equal(x,y)
    for x, y in zip(net.rcm.parameters(), r.trunk.parameters()): assert torch.equal(x,y)
    for x, y in zip(net.head.parameters(), h.head.parameters()): assert torch.equal(x,y)


@pytest.mark.parametrize('kind', ['junction', 'rcm'])
def test_training_exports_complete_model(tmp_path, monkeypatch, kind):
    torch.set_num_threads(2)
    rng = np.random.default_rng(9)
    a, d = rng.normal(size=(2, 20, 400, 512)).astype(np.float32)
    r = rng.normal(size=(20, 3, 5, 250)).astype(np.float32)
    y = np.tile([0, 1, 0, 1], 5)
    f = pd.DataFrame({'candidate_id':np.arange(20), 'y':y, 'fold':np.repeat(np.arange(5),4)})
    monkeypatch.setattr(training_cli.data, 'frame', lambda name:f)
    for key, value in {'max_epochs':1, 'min_epochs':1}.items(): monkeypatch.setitem(training.TRAIN,key,value)
    class Inputs:
        def __init__(self, *args, junction=None, **kwargs): self.y=y; self.junction=junction
        def use(self, *args): pass
        def batch(self, rows):
            rows=np.sort(rows)
            out={'acceptor':torch.from_numpy(a[rows]), 'donor':torch.from_numpy(d[rows]), 'rcm_raw':torch.from_numpy(r[rows])}
            return rows,out
    class Junction(Inputs):
        def batch(self, rows):
            rows,out=super().batch(rows); out.pop('rcm_raw'); return rows,out
    monkeypatch.setattr(training_cli, 'JunctionInputs', Junction)
    monkeypatch.setattr(training_cli, 'RCMInputs', Inputs)
    output=tmp_path/'trained'
    training_cli.train(kind, 'k562', 0, 42, output, 'cpu')
    cfg=json.loads((output/'config.json').read_text())
    net=CircPrio(pool='meanmax') if kind=='junction' else CircPrioRCM(pool='meanmax')
    net.load_state_dict(torch.load(output/'model.pt', weights_only=True), strict=True)
    assert cfg['class_weight']==1 and len(pd.read_csv(output/'test_predictions.csv'))==4
