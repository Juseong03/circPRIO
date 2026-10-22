"""circPRIO-RCM: junction representation fused with the intronic reverse-complementary-match (RCM) representation of CircCNNs.

The RCM descriptor (exact reverse-complementary k-mer matches, k = 5, 7, 9, 11, 13; across-flank and within-flank relations;
5 x 250 positional count maps over ten nested windows up to 3 kb) and its three-region convolutional encoder are those of
CircCNNs (Wang et al.; RCM_optuna_concate_10000), re-implemented here as `TriRCM` (legacy seqjcf.py). Published configuration
(CircPair-RCM, legacy version 'v4'): staged training — the mean-pooled junction trunk and the RCM trunk are trained with their own
heads, then frozen while only the fusion head on [junction 128; RCM 128] is trained. Parameter names match the legacy modules.
"""
import torch
from torch import nn

from .circprio import SiteEncoder


class RegionCNN(nn.Module):
    """CircCNNs RCM region CNN: (5, 250) -> Conv1d 5->128 (k5, s5) -> Conv1d 128->128 (k5, s5) -> flatten 1,280."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(5, 128, 5, stride=5), nn.BatchNorm1d(128), nn.ReLU(),
                                 nn.Conv1d(128, 128, 5, stride=5), nn.BatchNorm1d(128), nn.ReLU())

    def forward(self, x):
        return self.net(x).flatten(1)


class TriRCM(nn.Module):
    """Across-flank, within-upstream and within-downstream region CNNs on log1p counts (B, 3, 5, 250) -> 128."""

    def __init__(self, mask=None):
        super().__init__()
        self.regions = nn.ModuleList([RegionCNN() for _ in range(3)])
        self.fc = nn.Sequential(nn.Linear(3840, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(.3),
                                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(.3))
        if mask is not None:                       # descriptor ablation: fixed 0/1 mask (3, 5, 250), entries read as zero matches
            self.register_buffer('mask', torch.as_tensor(mask, dtype=torch.float32))
        else:
            self.mask = None

    def forward(self, rcm_raw, **_):
        if self.mask is not None:
            rcm_raw = rcm_raw * self.mask
        return self.fc(torch.cat([r(rcm_raw[:, i]) for i, r in enumerate(self.regions)], 1))


def fusion_head(width):
    return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128), nn.GELU(), nn.Dropout(.3), nn.Linear(128, 2))


class JunctionTrunk(nn.Module):
    """Junction branch of circPRIO-RCM: acceptor/donor encoders -> [2 x dim] -> Linear -> 128 (GELU, dropout 0.3).

    pool='mean' is the published branch (legacy JunctionTrunk/SideCNN); pool='meanmax' the legacy CircPairTrunk variant.
    """

    def __init__(self, pool='mean', width=512):
        super().__init__()
        self.upper, self.lower = SiteEncoder(width, pool), SiteEncoder(width, pool)
        self.fc = nn.Sequential(nn.Linear(2 * self.upper.dim, 128), nn.GELU(), nn.Dropout(.3))

    def forward(self, acceptor, donor, **_):
        return self.fc(torch.cat([self.upper(acceptor), self.lower(donor)], 1))


class Branch(nn.Module):
    """One trunk with its own head (stages 1 and 2 of staged training)."""

    def __init__(self, trunk):
        super().__init__()
        self.trunk, self.head = trunk, fusion_head(128)

    def forward(self, **inputs):
        return self.head(self.trunk(**inputs))


class GatedHead(nn.Module):
    """h = g * junction + (1 - g) * rcm with g = sigmoid(W [junction; rcm]); then LayerNorm -> 128 -> GELU -> dropout -> 2."""

    def __init__(self, d=128):
        super().__init__()
        self.gate = nn.Linear(2 * d, d)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.GELU(), nn.Dropout(.3), nn.Linear(128, 2))

    def forward(self, joined):
        s, r = joined.chunk(2, 1)
        g = torch.sigmoid(self.gate(joined))
        return self.out(g * s + (1 - g) * r)


class CircPrioRCM(nn.Module):
    """[junction 128; RCM 128] -> fusion head. Staged: load trained branch trunks, freeze them (eval mode), train the head."""

    def __init__(self, pool='mean', gated=False, junction=None):
        super().__init__()
        self.junction = junction if junction is not None else JunctionTrunk(pool)
        self.rcm = TriRCM()
        self.head = GatedHead() if gated else fusion_head(256)

    def forward(self, acceptor, donor, rcm_raw, **_):
        return self.head(torch.cat([self.junction(acceptor, donor), self.rcm(rcm_raw)], 1))

    def freeze_trunks(self):
        for trunk in (self.junction, self.rcm):
            trunk.requires_grad_(False)
        return self

    def train(self, mode=True):
        super().train(mode)
        for trunk in (self.junction, self.rcm):        # frozen trunks keep fixed BatchNorm statistics and no dropout
            if not any(p.requires_grad for p in trunk.parameters()):
                trunk.eval()
        return self


class FusionHead(nn.Module):
    """Staged fusion head alone, trained on cached frozen-trunk features [junction 128; RCM 128] (same function as CircPrioRCM
    with frozen trunks in eval mode, whose trunk outputs do not change during head training)."""

    def __init__(self, gated=False):
        super().__init__()
        self.head = GatedHead() if gated else fusion_head(256)

    def forward(self, features, **_):
        return self.head(features)
