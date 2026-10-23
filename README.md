# circPRIO

Circular RNA candidate prioritisation with **circPRIO** (junction sequence) and **circPRIO-RCM** (junction + reverse-complementary-match features).

This repository includes our model definitions, inference/training scripts, fixed candidate tables and compressed 400-nt sequence windows. **RCM inputs and the combined reproducibility package are available through the verified downloads below.** Model weights and learned embedding caches are excluded from Git. Comparator models and historical experiment code are excluded. Git LFS is not required for this source-and-small-data package.

## Data

| Asset | Location |
|---|---|
| Candidate tables: IDs, coordinates, labels, components, fixed folds | Included: `data/datasets/<dataset>/candidates.parquet` |
| Compressed 400-nt sequence windows and manifests | Included: `data/windows/<dataset>_L400.{npz,json}` |
| RCM descriptors for all five datasets | [Download RCM ZIP](https://drive.google.com/file/d/1ZO6Gfzje2SEwbXL-YE3iEPYAdter8Qzm/view) (372.84 MB); or regenerate with `scripts/make_rcm.py` |
| RCM flank lengths (intron-bounded datasets) | Included: `data/datasets/{k562,GM12878,HepG2}/rcm_flanks.parquet` |
| SpliceBERT weights | Download the pinned checkpoint with `scripts/download_splicebert.py` |
| SpliceBERT embeddings | Regenerate locally; ignored by Git |
| Trained circPRIO weights | Train locally or obtain separately from the project owner |

Dataset names: `circcnns`, `circ2lo`, `k562`, `GM12878`, `HepG2`. The first three have fixed training folds; the last two are transfer populations. See [data/README.md](data/README.md) for schemas and checksums.

For circPRIO, the dataset files already in this repository are sufficient to regenerate embeddings and train. For circPRIO-RCM, additionally regenerate RCM from a reference genome, or download and extract the RCM ZIP into the repository root. It creates `data/datasets/<dataset>/rcm.npy`. Do not commit these large arrays.

The RCM arrays can also be recomputed from sequence instead of downloaded (hg19 for `circcnns`/`circ2lo`, hg38 for the others; genomes are not distributed):

```bash
python scripts/make_rcm.py --dataset k562 --genome /path/to/hg38.fa --workers 16
```

The generator (`src/circprio/rcm_generate.py`) reproduces the distributed arrays exactly for every candidate of all five datasets; `--check <rcm.npy>` compares instead of writing.

The RCM ZIP is 372,839,187 bytes; its SHA-256 is `539aa1e97b420aa28d75191f0b342392cb4cb3acd760b9c1e5103ddac259c17a`. The complete remote file was downloaded without login and matched the original bytes. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for genome sources, verification commands and the distinction between plot reproduction and candidate-level statistical recomputation.

## Manuscript version and reproducibility scope

The manuscript cites the public source snapshot [`271dc008650b40fc901dbccd4a8a6b8a90997981`](https://github.com/Juseong03/circPRIO/tree/271dc008650b40fc901dbccd4a8a6b8a90997981). This snapshot was verified anonymously on 23 October 2026. It provides model training/inference code and fixed candidate inputs; it does not contain the saved paper predictions, trained checkpoints, or the manuscript figure pipeline. The current documentation/statistical-script revisions are local changes pending author review and publication; they are not part of that cited commit.

The [combined reproducibility ZIP](https://drive.google.com/file/d/15CJA3t0DxM6QxHfzx_79rC6ikqk8qWaS/view) (44.97 MB) contains `core_evidence/` (candidate-level K562 predictions and the independent statistical script) and `figure_reproduction/` (all seven main and two supplementary figures' saved inputs/code). It was downloaded without login and matched SHA-256 `a71400b0314b55cabc4ff1187e75813a19c08c440c045ad2d40bdf586a38f400`. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for download, extraction and exact commands. These external archives are separate from the original cited code commit; no durable DOI has yet been issued.

## Models

* **circPRIO**: frozen SpliceBERT states from jointly encoded 400-nt acceptor/donor windows, site-specific CNNs, mean+max pooling.
* **circPRIO-RCM**: separately trained junction and RCM trunks followed by frozen-trunk concatenation fusion (`staged_concat_meanmax`). The RCM descriptor and TriRCM encoder are from **CircCNNs** and are attributed accordingly.

Windows follow genomic coordinates in transcript orientation, not spliced transcripts. See `configs/window_policy.yaml`. Scores prioritise candidates under the training label definition; they are not evidence of a causal circularisation mechanism.

## Install

Python 3.10 and a compatible PyTorch installation are required. CPU inference is supported; embedding generation uses CUDA.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
# Match the original CUDA 12.1 environment, or install the PyTorch build for your machine.
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e . --no-deps
```

Use an editable installation: data/model paths resolve relative to the checkout. Mamba, comparator libraries and manuscript tools are not required.

## Train

```bash
python scripts/download_splicebert.py
python scripts/embed.py --dataset k562 --device cuda:0

python scripts/train.py --model junction --dataset k562 --fold 0 \
  --seed 20260910 --device cuda:0 --output outputs/junction/seed20260910/fold0

# First extract the RCM ZIP into this repository's root.
python scripts/train.py --model rcm --dataset k562 --fold 0 \
  --seed 20260910 --device cuda:0 --output outputs/rcm/seed20260910/fold0
```

Each command trains one fixed fold/seed. Repeat folds 0–4 and the desired seeds for cross-validation. Epoch selection uses the next fold, inner training uses three folds, and a fresh outer refit uses four folds. Seeds are set before model construction. RCM training fits both branches separately, builds phase-specific frozen representations and trains a concatenation head. No comparator or model-selection experiments are run.

Outputs include complete `model.pt`, companion `config.json`, training records and held-out predictions. Use a new empty output directory for each fit. `CIRCPRIO_SPLICEBERT=/path/to/local/checkpoint` overrides the frozen checkpoint location; its exact revision is recorded in `configs/model.json`.

## Run inference

Use a complete trained checkpoint and its companion config. Prepared inference `.npz` files must contain:

| Key | Shape | Meaning |
|---|---|---|
| `acceptor` | `(N, 400, 512)` | Frozen SpliceBERT acceptor states |
| `donor` | `(N, 400, 512)` | Donor states from the same joint encoding |
| `rcm` | `(N, 3, 5, 250)` | Log-transformed RCM descriptor; RCM model only |
| `candidate_id` | `(N,)`, optional | String/numeric IDs in feature order |

These are encoded features, not raw FASTA or the `windows` arrays committed in `data/windows/`. Preserve candidate alignment and use the pinned frozen checkpoint. Do not standardise RCM for TriRCM.

```bash
python scripts/predict.py \
  --inputs /path/to/features.npz \
  --checkpoint /path/to/model.pt \
  --config /path/to/config.json \
  --device cpu \
  --output outputs/predictions.csv
```

Use `--device cuda:0` for GPU inference. The CSV contains IDs, logits, prior-corrected scores and the class-balanced threshold. The config must specify `model` (`junction` or `rcm`), `pool` (`meanmax`) and the positive `class_weight` used in outer training. Do not substitute a new cohort's prevalence.

Historical RCM fits store two branch checkpoints and a fusion head separately. To assemble a complete checkpoint, provide files from the same dataset/seed/fold:

```bash
python scripts/export_checkpoint.py \
  --junction /path/to/junction_meanmax/SEED/fold0/outer_model.pt \
  --rcm /path/to/rcm_branch/SEED/fold0/outer_model.pt \
  --head /path/to/staged_concat_meanmax/SEED/fold0/outer_head.pt \
  --result /path/to/staged_concat_meanmax/SEED/fold0/result.json \
  --output checkpoints/rcm
```

## Layout

```text
src/circprio/          Our models, input loading, embeddings and training utilities
scripts/              Inference, checkpoint export, embedding and model training
configs/              Adopted model configuration and window convention
data/datasets/        Committed candidate tables; externally downloaded RCM arrays are ignored
data/windows/         Committed compressed sequence inputs and manifests
metadata/             Committed/external data sizes and checksums
tests/                Window, inference and training tests
```

Generated `artifacts/`, `checkpoints/` and `outputs/` are ignored by Git. Explicit `.gitignore` exceptions allow only the prepared candidate tables and compressed sequence windows to be committed.

## Tests

```bash
pip install pytest==9.1.1
python -m pytest -q
```

Execution tests use synthetic inputs without a GPU. Prepared-data tests verify included windows, hashes and fixed folds. They do not reproduce trained performance.

## Commit

```bash
git init
git add .
git status --short
git commit -m "Add circPRIO models and prepared small datasets"
```

Before publishing an external data download, confirm its source-data redistribution terms and record its tested URL. The code is released under the MIT License ([LICENSE](LICENSE)); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and third-party terms.

Archive provenance note: the downloaded ZIPs preserve README/manifest statements written before the author uploaded them (for example, “local review candidate” or “no public URL”). Those are historical preparation records. The verified URLs and hashes in `metadata/external_downloads.tsv` give the current technical access status; original scientific content and ZIP bytes have not been changed.
