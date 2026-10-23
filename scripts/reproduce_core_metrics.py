"""Recompute the frozen K562 core comparison without training or the legacy workspace.

Input: a separately supplied core-evidence directory, not summary CSVs alone.
This script reads candidate-level float64 scores and verifies all recorded hashes.
Bootstrap estimates condition on the original predictions; no models are refitted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange, set_num_threads
from sklearn.metrics import average_precision_score, roc_auc_score


@njit(parallel=True)
def weighted_metrics(y, components, last, weights, positives, negatives):
    """Tie-aware weighted AUROC/AP in a fixed descending-score order."""
    out = np.empty((len(weights), 2), np.float64)
    for b in prange(len(weights)):
        tp = 0.0
        fp = 0.0
        prev_tp = 0.0
        prev_fp = 0.0
        ap = 0.0
        missed_area = 0.0
        for i in range(len(y)):
            w = weights[b, components[i]]
            tp += w * y[i]
            fp += w * (1 - y[i])
            if last[i]:
                if tp + fp > 0:
                    ap += (tp - prev_tp) * tp / (tp + fp)
                missed_area += (fp - prev_fp) * (positives[b] - tp + (tp - prev_tp) / 2)
                prev_tp = tp
                prev_fp = fp
        out[b, 0] = 1 - missed_area / (positives[b] * negatives[b])
        out[b, 1] = ap / positives[b]
    return out


def component_draws(codes, y, repeats, seed):
    rng = np.random.default_rng(seed)
    n = int(codes.max()) + 1
    positive = np.bincount(codes, weights=y, minlength=n)
    negative = np.bincount(codes, weights=1-y, minlength=n)
    draws, rejected = [], 0
    while len(draws) < repeats:
        w = np.bincount(rng.integers(n, size=n), minlength=n)
        if w @ positive == 0 or w @ negative == 0:
            rejected += 1
            continue
        draws.append(w)
    weights = np.asarray(draws, dtype=np.int64)
    return weights, weights @ positive, weights @ negative, rejected


def gene_clusters(g):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for gene, comp in zip(g.gene_id, 'c' + g.component.astype(str)):
        a, b = find(gene), find(comp)
        if a != b:
            parent[a] = b
    return {gene: find(gene) for gene in g.gene_id.unique()}


def hit_per_gene(g, scores):
    d = g[['candidate_id', 'gene_id', 'y']].copy()
    d['score'] = scores
    grouped = d.groupby('gene_id').y.agg(['sum', 'size'])
    eligible = grouped.index[(grouped['sum'] > 0) & (grouped['sum'] < grouped['size'])]
    ranked = d.sort_values(['gene_id', 'score', 'candidate_id'], ascending=[True, False, True], kind='mergesort')
    return ranked.drop_duplicates('gene_id').set_index('gene_id').y.reindex(sorted(eligible)).astype(float)


def run(folder, output):
    manifest = json.loads((folder/'manifest.json').read_text())
    for item in manifest['files']:
        assert hashlib.sha256((folder/item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
    set_num_threads(min(4, os.cpu_count() or 1))
    frame = pd.read_csv(folder/'candidates.tsv.gz', sep='\t', keep_default_na=False)
    scores = np.load(folder/'scores.npz', allow_pickle=False)
    genes = pd.read_csv(folder/'within_gene_candidates.tsv.gz', sep='\t')
    assert frame.candidate_id.is_unique and genes.candidate_id.is_unique
    order = pd.Series(np.arange(len(frame)), index=frame.candidate_id).loc[genes.candidate_id].to_numpy()
    np.testing.assert_array_equal(genes.y, frame.y.to_numpy()[order])
    np.testing.assert_array_equal(genes.component, frame.component.to_numpy()[order])
    y = frame.y.to_numpy(np.int64)
    codes = pd.factorize(frame.component)[0].astype(np.int64)
    weights, positive, negative, rejected = component_draws(codes, y, manifest['bootstrap']['repeats'], manifest['bootstrap']['seed'])
    point, boot, gene_points = [], {}, {}
    for model, key in manifest['models'].items():
        s = scores[key]
        assert s.shape == (len(manifest['seeds']), len(frame)) and np.isfinite(s).all()
        per, draws, hits = [], [], []
        for vector in s:
            per.append([roc_auc_score(y, vector), average_precision_score(y, vector)])
            ix = np.argsort(-vector, kind='mergesort')
            last = np.r_[vector[ix][1:] != vector[ix][:-1], True]
            draws.append(weighted_metrics(y[ix], codes[ix], last, weights, positive, negative))
            hits.append(hit_per_gene(genes, vector[order]))
        per = np.asarray(per)
        boot[model] = np.mean(draws, axis=0)
        gene_points[model] = sum(hits)/len(hits)
        point.append(dict(model=model, AP=per[:, 1].mean(), AUROC=per[:, 0].mean(), Hit_at_1=gene_points[model].mean(),
                          AP_ci_low=np.percentile(boot[model][:, 1], 2.5), AP_ci_high=np.percentile(boot[model][:, 1], 97.5)))
        print(model, 'AP', point[-1]['AP'], 'Hit@1', point[-1]['Hit_at_1'], flush=True)
    point = pd.DataFrame(point).set_index('model')
    clusters = gene_clusters(genes)
    eligible = gene_points[next(iter(gene_points))].index
    gene_codes = pd.factorize(pd.Series([clusters[x] for x in eligible]))[0]
    n = gene_codes.max()+1
    rng = np.random.default_rng(manifest['bootstrap']['seed'])
    W = np.array([np.bincount(rng.integers(n, size=n), minlength=n)[gene_codes] for _ in range(manifest['bootstrap']['repeats'])], dtype=float)
    differences = []
    for a, b in manifest['contrasts']:
        for j, metric in enumerate(['AUROC', 'AP']):
            v = boot[a][:, j]-boot[b][:, j]
            differences.append(dict(model=a, comparator=b, metric=metric, delta=point.loc[a,metric]-point.loc[b,metric],
                                    ci_low=np.percentile(v,2.5), ci_high=np.percentile(v,97.5)))
        v = W @ (gene_points[a]-gene_points[b]).to_numpy() / W.sum(axis=1)
        differences.append(dict(model=a, comparator=b, metric='Hit_at_1', delta=point.loc[a,'Hit_at_1']-point.loc[b,'Hit_at_1'],
                                ci_low=np.percentile(v,2.5), ci_high=np.percentile(v,97.5)))
    differences = pd.DataFrame(differences)
    ref = pd.read_csv(folder/'expected_summary.tsv',sep='\t').set_index('model')
    refwg = pd.read_csv(folder/'expected_within_gene_summary.tsv',sep='\t').set_index('model')
    checks = []
    for model in point.index:
        for metric in ['AP','AUROC','AP_ci_low','AP_ci_high','Hit_at_1']:
            expected = (refwg if metric=='Hit_at_1' else ref).loc[model,metric]
            error = abs(point.loc[model,metric]-expected)
            checks.append(dict(model=model,metric=metric,error=float(error),passed=bool(error<1e-10)))
    refd = pd.concat([pd.read_csv(folder/f,sep='\t') for f in ['expected_deltas.tsv','expected_within_gene_deltas.tsv']])
    for r in differences.itertuples(index=False):
        expected = refd[(refd.model==r.model)&(refd.comparator==r.comparator)&(refd.metric==r.metric)].iloc[0]
        for field in ['delta','ci_low','ci_high']:
            error = abs(getattr(r,field)-expected[field])
            checks.append(dict(model=r.model,comparator=r.comparator,metric=r.metric,field=field,error=float(error),passed=bool(error<1e-10)))
    output.mkdir(parents=True,exist_ok=True)
    point.to_csv(output/'recomputed_summary.tsv',sep='\t')
    differences.to_csv(output/'recomputed_deltas.tsv',sep='\t',index=False)
    report=dict(status='PASSED' if all(c['passed'] for c in checks) else 'FAILED',
                candidate_rows=len(frame), positives=int(y.sum()), components=len(set(codes)),seeds=manifest['seeds'],
                repeats=manifest['bootstrap']['repeats'],bootstrap_seed=manifest['bootstrap']['seed'],
                rejected_single_class_draws=rejected,within_gene_candidates=len(genes),eligible_genes=len(eligible),gene_clusters=int(n),checks=checks)
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    assert report['status']=='PASSED', 'Recomputed values differ: inspect verification.json; manuscript values were not changed.'
    print('All',len(checks),'source-value checks passed.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    run(a.evidence,a.output)
