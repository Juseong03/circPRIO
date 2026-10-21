# circPRIO

Circular RNA candidate prioritisation with **circPRIO-J** (junction sequence) and **circPRIO-RCM** (junction + reverse-complementary-match features).

This repository includes our model definitions, inference/training scripts, fixed candidate tables and compressed 400-nt sequence windows. **Only the large RCM descriptors are distributed through Google Drive.** Model weights and learned embedding caches are excluded from Git. Comparator models and historical experiment code are excluded. Git LFS is not required for this source-and-small-data package.

## Data

| Asset | Location |
|---|---|
| Candidate tables: IDs, coordinates, labels, components, fixed folds | Included: `data/datasets/<dataset>/candidates.parquet` |
| Compressed 400-nt sequence windows and manifests | Included: `data/windows/<dataset>_L400.{npz,json}` |
| RCM descriptors for all five datasets | `circPRIO_RCM_20261021.zip`: **Google Drive link to be added** |
| SpliceBERT weights | Download the pinned checkpoint with `scripts/download_splicebert.py` |
| SpliceBERT embeddings | Regenerate locally; ignored by Git |
| Trained circPRIO weights | Train locally or obtain separately from the project owner |

Dataset names: `circcnns`, `circ2lo`, `k562`, `GM12878`, `HepG2`. The first three have fixed training folds; the last two are transfer populations. See [data/README.md](data/README.md) for schemas and checksums.

For circPRIO-J, the dataset files already in this repository are sufficient to regenerate embeddings and train. For circPRIO-RCM, additionally download the RCM ZIP and extract into the repository root. It creates `data/datasets/<dataset>/rcm.npy`. Do not commit these large arrays.

The original working project contains the RCM ZIP under `release/downloads/`; upload this ZIP to Drive and replace the link placeholder above. No upload has been performed automatically.

## Models

* **circPRIO-J**: frozen SpliceBERT states from jointly encoded 400-nt acceptor/donor windows, site-specific CNNs, mean+max pooling.
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

Replace the RCM Drive link before publishing. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and licensing status.
