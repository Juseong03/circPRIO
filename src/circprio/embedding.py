"""Frozen SpliceBERT states of splice-site windows, cached per (data set, window length, encoding mode, checkpoint).

mode='joint'    : acceptor and donor windows concatenated into one input (2L nt) and embedded in one pass (historical CircPair);
                  states split back into acceptor [:L] and donor [L:].
mode='separate' : each window embedded on its own (L nt); acceptor and donor never exchange information inside SpliceBERT.
Both: multimolecule SpliceBERT (hidden 512, max 1,024 nt), fp16 autocast inference, [CLS]/[SEP] states removed, fp16 storage,
identical inputs embedded once (unique table + index), exactly as the legacy embed()/embed_separate().

Cache layout: artifacts/cache/splicebert/<dataset>/L<length>_<mode>/{states.npy, ids.npy, manifest.json}
  joint    states (U, 2L, 512), ids (N,)      separate  states (U, L, 512), ids (N, 2) [acceptor, donor]
The manifest records the window-table hash, checkpoint hash, mode and length; `load` refuses a cache whose keys do not match.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import ARTIFACTS
from .data import WINDOW_CACHE, windows

CACHE = ARTIFACTS / 'cache/splicebert'
import os
SNAPSHOT = Path(os.environ.get('CIRCPRIO_SPLICEBERT', str(ARTIFACTS / 'models/splicebert')))

BASES = np.array(list('ACGT'))
MAX_NT = 1024


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def splicebert(device):
    import multimolecule
    from safetensors.torch import load_file
    tokenizer = multimolecule.RnaTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    network = multimolecule.SpliceBertModel(multimolecule.SpliceBertConfig.from_pretrained(SNAPSHOT, local_files_only=True),
                                            add_pooling_layer=False)
    weights = load_file(str(SNAPSHOT / 'model.safetensors'))
    network.load_state_dict({k.removeprefix('model.'): v for k, v in weights.items() if k.startswith('model.')}, strict=True)
    return tokenizer, network.requires_grad_(False).eval().to(device), sha_file(SNAPSHOT / 'model.safetensors')


def encode_batch(tokenizer, network, seqs, width, device):
    """Equal-length sequences -> fp16 states (B, width, 512) without [CLS]/[SEP] (legacy procedure)."""
    inputs = tokenizer(list(seqs), return_tensors='pt', padding=False, truncation=False, add_special_tokens=True)
    names = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist())
    if inputs['input_ids'].shape[1] != width + 2 or ''.join(names[1:-1]).upper().replace('U', 'T') != seqs[0]:
        raise ValueError('SpliceBERT token alignment error')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        h = network(**{k: v.to(device) for k, v in inputs.items()}).last_hidden_state.float()
    if not torch.isfinite(h).all():
        raise ValueError('nonfinite SpliceBERT states')
    return h[:, 1:-1].cpu().numpy().astype(np.float16)


def folder(name, length, mode):
    return CACHE / name / f'L{length}_{mode}'


def embed(name, length, mode, device='cuda:0', batch=64):
    if mode not in ('joint', 'separate'):
        raise ValueError('mode must be joint or separate')
    out = folder(name, length, mode)
    if (out / 'manifest.json').exists():
        return out
    w = np.asarray(windows(name, length))
    valid = (w != 255).all(axis=(1, 2))
    seqs = np.array([''.join(BASES[np.where(r == 255, 0, r)]) for r in (w.reshape(len(w), -1) if mode == 'joint' else w.reshape(-1, length))])
    width = 2 * length if mode == 'joint' else length
    if width > MAX_NT:
        raise ValueError(f'{width} nt exceeds the SpliceBERT limit of {MAX_NT}')
    unique, inverse = np.unique(seqs, return_inverse=True)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer, network, checkpoint = splicebert(torch.device(device))
    states = np.lib.format.open_memmap(out / 'states.npy', mode='w+', dtype=np.float16, shape=(len(unique), width, 512))
    for start in range(0, len(unique), batch):
        chunk = list(unique[start:start + batch])
        states[start:start + len(chunk)] = encode_batch(tokenizer, network, chunk, width, device)
    states.flush()
    ids = inverse.astype(np.int64) if mode == 'joint' else inverse.reshape(-1, 2).astype(np.int64)
    np.save(out / 'ids.npy', ids)
    manifest = json.loads((WINDOW_CACHE / f'{name}_L{length}.json').read_text())
    (out / 'manifest.json').write_text(json.dumps(dict(
        dataset=name, length=length, mode=mode, rows=len(w), unique_inputs=int(len(unique)), invalid_rows=int((~valid).sum()),
        windows_sha256=manifest['sha256'], checkpoint_sha256=checkpoint, checkpoint=str(SNAPSHOT.name), precision='fp16 autocast',
        layout='joint: (U, 2L, 512) acceptor[:L] donor[L:]; ids (N,)' if mode == 'joint' else 'separate: (U, L, 512); ids (N, 2)'),
        indent=2) + '\n')
    return out


class States:
    """Random access to cached states: acceptor, donor arrays (B, L, 512) for row indices."""

    def __init__(self, name, length, mode):
        d = folder(name, length, mode)
        self.manifest = json.loads((d / 'manifest.json').read_text())
        w = json.loads((WINDOW_CACHE / f'{name}_L{length}.json').read_text())
        if (self.manifest['dataset'], self.manifest['length'], self.manifest['mode'], self.manifest['windows_sha256']) != (name, length, mode, w['sha256']):
            raise ValueError('embedding cache does not match the requested data set, length, mode or windows')
        self.length, self.mode = length, mode
        self.states = np.load(d / 'states.npy', mmap_mode='r')
        self.ids = np.load(d / 'ids.npy')

    def get(self, rows):
        rows = np.asarray(rows)
        if self.mode == 'joint':
            s = np.asarray(self.states[self.ids[rows]])
            return s[:, :self.length], s[:, self.length:]
        return np.asarray(self.states[self.ids[rows, 0]]), np.asarray(self.states[self.ids[rows, 1]])
