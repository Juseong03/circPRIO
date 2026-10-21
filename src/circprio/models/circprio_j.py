"""circPRIO-J: junction model on frozen SpliceBERT states of the acceptor and donor windows.

Historical configuration (published as CircPair; legacy `ablate.py` AblationNet 'p_meanmax'): the joined 400-nt input is
embedded once by SpliceBERT (joint encoding), split into acceptor and donor states (2 x 200 x 512), each projected 512->128 (GELU),
two kernel-9 convolutions (128 channels, GELU), mean+max pooling (2 x 256), concatenation (512), LayerNorm->128->GELU->
Dropout(0.3)->2. Parameter names are identical to the legacy modules so that legacy checkpoints load with strict=True.

The same class covers the legacy component ablations through its arguments (pooling, shared encoder, single window, one-hot input).
"""
import torch
from torch import nn
from torch.nn import functional as F

POOLINGS = ('mean', 'max', 'meanmax', 'attn', 'topk')


class SiteEncoder(nn.Module):
    """One splice-site window (B, L, width) -> pooled vector. Legacy name: Side."""

    def __init__(self, width=512, pool='meanmax', hidden=128, kernel=9):
        super().__init__()
        if pool not in POOLINGS:
            raise ValueError(f'pool must be one of {POOLINGS}')
        self.pool = pool
        self.project = nn.Sequential(nn.Linear(width, hidden), nn.GELU())
        self.conv = nn.Sequential(nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2), nn.GELU(),
                                  nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2), nn.GELU())
        self.dim = 2 * hidden if pool == 'meanmax' else hidden
        if pool == 'attn':
            self.score = nn.Conv1d(hidden, 1, 1)

    def features(self, x):
        """Position-wise features (B, hidden, L) before pooling."""
        return self.conv(self.project(x).transpose(1, 2))

    def forward(self, x):
        h = self.features(x)
        if self.pool == 'max':
            return h.amax(-1)
        if self.pool == 'meanmax':
            return torch.cat([h.mean(-1), h.amax(-1)], 1)
        if self.pool == 'attn':
            return (h * torch.softmax(self.score(h), -1)).sum(-1)
        if self.pool == 'topk':
            return h.topk(10, -1).values.mean(-1)
        return h.mean(-1)


def classifier_head(width):
    return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128), nn.GELU(), nn.Dropout(.3), nn.Linear(128, 2))


class CircPrioJ(nn.Module):
    """circPRIO-J (historical CNN encoder).

    Inputs: `acceptor` and `donor` states (B, L, width) — from joint encoding (split of one SpliceBERT pass) or separate encoding
    (two passes); the model is agnostic to which. `sites`: 'both', 'acceptor' or 'donor'. `shared`: one encoder for both sites.
    `width=4` with one-hot input reproduces the 'no SpliceBERT' ablation.
    """

    def __init__(self, pool='meanmax', width=512, shared=False, sites='both'):
        super().__init__()
        if sites not in ('both', 'acceptor', 'donor'):
            raise ValueError('sites must be both, acceptor or donor')
        self.sites = sites
        self.upper = SiteEncoder(width, pool)                         # acceptor (legacy 'upper')
        self.lower = self.upper if shared else SiteEncoder(width, pool)   # donor (legacy 'lower')
        d = self.upper.dim if sites != 'both' else 2 * self.upper.dim
        self.head = classifier_head(d)

    def encode(self, acceptor, donor):
        if self.sites == 'acceptor':
            return self.upper(acceptor)
        if self.sites == 'donor':
            return self.lower(donor)
        return torch.cat([self.upper(acceptor), self.lower(donor)], 1)

    def forward(self, acceptor, donor):
        return self.head(self.encode(acceptor, donor))

    @staticmethod
    def split_joint(states, length=200):
        """Legacy joint states (B, 2*length, width) -> (acceptor, donor)."""
        return states[:, :length], states[:, length:]

    @staticmethod
    def one_hot(tokens):
        """ACGT codes (B, 2, L) -> one-hot (acceptor, donor), each (B, L, 4)."""
        oh = F.one_hot(tokens.long(), 4).float()
        return oh[:, 0], oh[:, 1]


LEGACY_VARIANTS = {          # legacy ablate.py version -> constructor arguments (states as in the legacy runs)
    'p_meanmax': dict(pool='meanmax'), 'v1': dict(pool='mean'), 'a_maxpool': dict(pool='max'), 'p_attn': dict(pool='attn'),
    'p_topk': dict(pool='topk'), 'a_shared': dict(pool='mean', shared=True), 'a_upper': dict(pool='mean', sites='acceptor'),
    'a_lower': dict(pool='mean', sites='donor'), 'a_onehot': dict(pool='mean', width=4), 'a_separate': dict(pool='mean')}
