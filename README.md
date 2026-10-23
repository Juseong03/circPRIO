# circPRIO

circPRIO prioritises circular RNA back-splice candidates using jointly encoded, pretrained acceptor/donor sequence representations. circPRIO-RCM combines the junction branch with the published CircCNNs intronic reverse-complementary-match (RCM) descriptor, encoder and staged fusion.

Scores rank candidates under the training label definition; they are not calibrated circularisation probabilities, abundance estimates or evidence of a causal mechanism. Windows are genomic, transcript-oriented sequences, not contiguous spliced RNA. See `configs/window_policy.yaml`.

## Installation

Use Python 3.10 and a compatible PyTorch build. The original environment used CUDA 12.1:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e . --no-deps
```

CPU inference is supported; SpliceBERT embedding generation uses CUDA. Mamba and comparator libraries are not needed.

## Data and downloads

Candidate tables, fixed folds, compressed 400-nt windows and cell-line flank lengths are included for `circcnns`, `circ2lo`, `k562`, `GM12878` and `HepG2`. The first three are training/evaluation populations; the last two are frozen-transfer populations. Schemas and array alignment are documented in [data/README.md](data/README.md).

| Download | Size | Contents |
|---|---:|---|
| [RCM data](https://drive.google.com/file/d/1ZO6Gfzje2SEwbXL-YE3iEPYAdter8Qzm/view) | 372.84 MB | RCM descriptors for all five populations |
| [Reproducibility package](https://drive.google.com/file/d/15CJA3t0DxM6QxHfzx_79rC6ikqk8qWaS/view) | 44.97 MB | K562 candidate-level predictions/statistical code and saved inputs/code for Figures 1–7 and S1–S2 |

Download, verify and extract from the repository root:

```bash
curl --fail --location 'https://drive.usercontent.google.com/download?id=1ZO6Gfzje2SEwbXL-YE3iEPYAdter8Qzm&export=download&confirm=t' --output circPRIO_RCM_20261021.zip
curl --fail --location 'https://drive.usercontent.google.com/download?id=15CJA3t0DxM6QxHfzx_79rC6ikqk8qWaS&export=download&confirm=t' --output circPRIO_reproducibility_20261023.zip
sha256sum -c metadata/external_download_SHA256SUMS
unzip circPRIO_RCM_20261021.zip
sha256sum -c data/rcm_SHA256SUMS
unzip circPRIO_reproducibility_20261023.zip -d reproducibility
```

Both full downloads were verified without login against the original SHA-256 values and ZIP CRCs. Exact URLs, sizes and hashes are in `metadata/external_downloads.tsv`; included data-file hashes are in `metadata/github_data_files.tsv` and RCM array hashes in `metadata/rcm_files.tsv`. Large RCM arrays, weights and embedding caches are excluded from Git. Raw RNA-seq and reference genomes are not distributed here.

### Regenerating RCM

RCM can be regenerated instead of downloaded:

```bash
python scripts/make_rcm.py --dataset k562 --genome /path/to/hg38.fa --workers 16
python scripts/make_rcm.py --dataset k562 --genome /path/to/hg38.fa --check data/datasets/k562/rcm.npy --workers 16
```

Use [hg19](https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz) for `circcnns`/`circ2lo` and [hg38](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz) for the cell-line populations. Preserve the provided intron-length tables, chromosome names and candidate order. The study FASTA SHA-256 values are `4029ee1557640ca5503abb84af511c45ce2d902cb13579a4c1234ae05afbfc62` (hg19) and `3bb6151693a3b598d510d8e43b1600557ed756e7cfc9e150fb94d1d9e6b3dd07` (hg38); file formatting can affect these hashes. Do not standardise RCM descriptors for TriRCM.

## Training

```bash
python scripts/download_splicebert.py
python scripts/embed.py --dataset k562 --device cuda:0
python scripts/train.py --model junction --dataset k562 --fold 0 --seed 20260910 --device cuda:0 --output outputs/junction/seed20260910/fold0
python scripts/train.py --model rcm --dataset k562 --fold 0 --seed 20260910 --device cuda:0 --output outputs/rcm/seed20260910/fold0
```

Each command fits one fold/seed; repeat folds 0–4 and seeds 20260910–20260919 for the ten-seed protocol. Inner training uses three folds, the next fold selects the epoch, and a fresh outer refit uses four folds; seeds are set before model construction. RCM training separately fits both trunks and then the fusion head on their phase-specific frozen features. Use a new empty output directory for each fit. Outputs include `model.pt`, `config.json`, training records and test predictions.

The frozen SpliceBERT checkpoint is pinned in `configs/model.json`; `CIRCPRIO_SPLICEBERT=/path/to/checkpoint` overrides its local location. No published trained circPRIO checkpoint is bundled, so inference requires a separately obtained or locally trained model. The compact repository does not rerun comparator/model-selection experiments.

## Inference

A prepared feature NPZ contains `acceptor` and `donor`, each `(N,400,512)` from the same joint SpliceBERT encoding; RCM models additionally require `rcm` of shape `(N,3,5,250)`. Optional `candidate_id` has shape `(N,)`. These encoded features differ from the raw compressed nucleotide windows in `data/windows/`.

```bash
python scripts/predict.py --inputs /path/to/features.npz --checkpoint /path/to/model.pt --config /path/to/config.json --device cpu --output outputs/predictions.csv
```

The companion config records `model` (`junction` or `rcm`), `pool` (`meanmax`) and the original positive `class_weight`. Do not replace that weight with inference-cohort prevalence. Output includes IDs, logits, prior-corrected scores and the training-derived class-balanced threshold. To assemble historical branch/head files from the same fit:

```bash
python scripts/export_checkpoint.py --junction /path/to/junction/outer_model.pt --rcm /path/to/rcm/outer_model.pt --head /path/to/fusion/outer_head.pt --result /path/to/fusion/result.json --output checkpoints/rcm
```

## Reproducing the reported results

From the repository root after extracting the combined ZIP:

```bash
python scripts/reproduce_core_metrics.py --evidence reproducibility/core_evidence --output outputs/core_recomputed
```

This independently recalculates AP/AUROC, Hit@1 and paired 2,000-draw component/gene-cluster intervals from unchanged K562 scores for circPRIO-RCM, the CircCNNs RCM model, the RCM branch and the same-architecture one-hot fusion (ten seeds, five folds). All 47 comparisons against the saved results passed at tolerance 1e-10. The executed bootstrap seed was 20261019; the preserved M8 configuration's prose says 20261010, but its analysis calls used the evaluation routine's default 20261019. Reproduction follows the executed code and leaves the original results unchanged.

For figure reproduction, install `reproducibility/figure_reproduction/reproduction/requirements.txt` in a separate Python environment, then run from the repository root:

```bash
python reproducibility/figure_reproduction/reproduction/manuscript/render_figures.py --figures 1 2 3 4 5 6 7 --supplementary --output outputs/replotted_figures
```

This second command plots saved summaries; it does not independently recalculate every metric/CI. Candidate-level recalculation covers the four K562 models above; other benchmark/transfer results are supplied as summaries. The archives contain no raw reads, genomes, trained checkpoints or large embedding caches. Pre-upload README/manifest status statements in the frozen ZIPs are historical; the download locations above are current.

The original study-code snapshot is [271dc008650b40fc901dbccd4a8a6b8a90997981](https://github.com/Juseong03/circPRIO/tree/271dc008650b40fc901dbccd4a8a6b8a90997981); later statistical scripts and download documentation are provided in this repository. Scores and bootstrap intervals remain conditional on fixed predictions and the study's candidate definitions. The K562 junction increment was not established in the evaluated frozen external transfer; pooled/non-fully-nested selection and comparator fitting differences remain limitations described in the paper.

## Tests and licence

```bash
pip install pytest==9.1.1
python -m pytest -q
```

Tests cover synthetic inference/training, RCM generation rules and prepared-data hashes/folds. They do not reproduce all trained-model performance. Project code uses the [MIT licence](LICENSE); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for inherited components and source-data terms.
