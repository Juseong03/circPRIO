# Reproducibility and release scope

Checked locally on 23 October 2026. No new paper model training was performed. Local documentation changes and the new statistical reproduction script have not been pushed.

## Fixed source and included inputs

The manuscript source version is `271dc008650b40fc901dbccd4a8a6b8a90997981`:

```
git clone https://github.com/Juseong03/circPRIO.git
cd circPRIO
git checkout 271dc008650b40fc901dbccd4a8a6b8a90997981
```

That public version includes candidate IDs, labels, coordinates, component/fold IDs, compressed 400-nt windows, intron flank lengths, configuration and model/training/inference scripts. It excludes paper predictions, trained checkpoints, backbone weights, embedding caches, comparator implementations and the paper's figure pipeline. Its 26 tests passed on an anonymously downloaded source copy in the existing Python 3.10/PyTorch environment; a fresh dependency installation and full GPU embedding/training workflow were NOT VERIFIED. Existing execution tests fit only tiny synthetic test models, not paper experiments.

Install and training/inference examples are in README. Training one model is different from regenerating all paper experiments. Model selection and comparator adaptations require the original documented study protocols; the compact repository intentionally excludes comparator code.

## RCM input: download status and genome regeneration

Public RCM: [share page](https://drive.google.com/file/d/1ZO6Gfzje2SEwbXL-YE3iEPYAdter8Qzm/view) · [direct download](https://drive.usercontent.google.com/download?id=1ZO6Gfzje2SEwbXL-YE3iEPYAdter8Qzm&export=download&confirm=t). Public combined reproducibility package: [share page](https://drive.google.com/file/d/15CJA3t0DxM6QxHfzx_79rC6ikqk8qWaS/view) · [direct download](https://drive.usercontent.google.com/download?id=15CJA3t0DxM6QxHfzx_79rC6ikqk8qWaS&export=download&confirm=t). Both complete archives were downloaded without any account/token and their SHA-256 values matched the original ZIPs; ZIP CRCs also passed. `metadata/external_downloads.tsv` records exact sizes, URLs and hashes. The author uploaded these files; no agent upload occurred. Source-data permission review remains separate from technical accessibility (THIRD_PARTY_NOTICES.md).

Download with the canonical local filenames below; the share-page filename may be shortened without changing the content:

```bash
curl --fail --location 'https://drive.usercontent.google.com/download?id=1ZO6Gfzje2SEwbXL-YE3iEPYAdter8Qzm&export=download&confirm=t' --output circPRIO_RCM_20261021.zip
curl --fail --location 'https://drive.usercontent.google.com/download?id=15CJA3t0DxM6QxHfzx_79rC6ikqk8qWaS&export=download&confirm=t' --output circPRIO_reproducibility_20261023.zip
sha256sum -c metadata/external_download_SHA256SUMS
```

When the archive is obtained, verify it before extraction:

```bash
printf '%s  %s\n' 539aa1e97b420aa28d75191f0b342392cb4cb3acd760b9c1e5103ddac259c17a circPRIO_RCM_20261021.zip | sha256sum -c -
unzip circPRIO_RCM_20261021.zip
sha256sum -c data/rcm_SHA256SUMS
```

It contains five float32 `(N,3,5,250)` descriptor arrays aligned to the committed candidate rows, plus per-array checksums. These are input descriptors, not learned weights. All five archived arrays were fully read, their ZIP CRCs verified and SHA-256 values matched `metadata/rcm_files.tsv` in the local submission audit. Anonymous remote download returned the identical original ZIP (not an HTML preview or login page).

Genomes can be obtained from UCSC assembly distributions:

- hg19: https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
- hg38: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz

Check the assembly README and applicable track/source terms. The exact local uncompressed FASTA checksums used by the study are `4029ee1557640ca5503abb84af511c45ce2d902cb13579a4c1234ae05afbfc62` (hg19) and `3bb6151693a3b598d510d8e43b1600557ed756e7cfc9e150fb94d1d9e6b3dd07` (hg38). Formatting/contig ordering can change whole-file hashes, so an assembly label alone does not prove byte identity. The current remote FASTA bytes have not been redownloaded in this audit.

```
python scripts/make_rcm.py --dataset k562 --genome /path/to/hg38.fa --workers 16
python scripts/make_rcm.py --dataset k562 --genome /path/to/hg38.fa --check data/datasets/k562/rcm.npy --workers 16
```

Repeat for circcnns/circ2lo with hg19 and GM12878/HepG2 with hg38. The committed flank-length tables preserve the cell-line intron limits. The published earlier all-candidate regeneration audit is preserved; this preflight does not rerun that expensive full computation.

## Candidate-level core metrics and intervals

The verified combined ZIP expands to `core_evidence/` and `figure_reproduction/` in the extraction directory. `core_evidence/` contains unchanged float64 K562 OOF scores for circPRIO-RCM, CircCNNs RCM model, RCM branch and same-architecture one-hot fusion (ten seeds/five folds), candidate labels/components/folds, within-gene assignments, expected results and provenance. It excludes raw reads, genomes and checkpoints. It is separate from the original code commit and currently has no persistent DOI.

From the directory containing the downloaded ZIP:

```bash
mkdir -p reproducibility
unzip circPRIO_reproducibility_20261023.zip -d reproducibility
cd reproducibility
sha256sum -c SHA256SUMS
python core_evidence/reproduce_core_metrics.py --evidence core_evidence --output recomputed_core
```

A copy of the script is inside the archive. It requires numpy, pandas, scikit-learn and numba. The executed analysis used 2,000 draws and seed **20261019**. The original M8 YAML prose says 20261010, but the analysis called the evaluation routine without overriding its default 20261019; the script follows the executed analysis and exactly reproduces the saved intervals. This metadata inconsistency does not change results and is not silently “corrected” by rerunning at another seed.

Checks reproduced AP 0.2121650965883835 (circPRIO-RCM) versus 0.1919677200118263 (CircCNNs RCM model), paired AP +0.0201973766 [0.0133260195,0.0270389542]; pretrained versus one-hot fusion +0.0322221906 [0.0265480229,0.0380541611]. Hit@1 and its paired intervals also reproduced; 47 comparisons passed at tolerance 1e-10. Bootstrap estimates condition on fixed predictions and do not account for refitting/selection uncertainty. External negative transfer findings are unchanged and not re-estimated by this K562-only archive.

## Figure reproduction is a separate task

The combined ZIP's `figure_reproduction/` directory contains Additional file 2's saved summaries, CSVs and standalone plotting source. Install `figure_reproduction/reproduction/requirements.txt` in a separate Python environment. From the extracted combined ZIP root:

```bash
python figure_reproduction/reproduction/manuscript/render_figures.py --figures 1 2 3 4 5 6 7 --supplementary --output replotted_figures
```

If using the journal's separate Additional file 2 instead, omit the leading `figure_reproduction/` directory from those paths.

This plotting command does not recompute predictions, metrics or CIs. The earlier isolated run regenerated all seven main and two supplementary figures. The core evidence adds true candidate-level recalculation only for the selected K562 comparisons. Other models, matched/interpretation analyses and external predictions require their own archived inputs; no claim of complete end-to-end reproduction is made.

## Review/release boundary

Original seed-initialisation differences, non-fully-nested selection, putative negatives, previously inspected external cohorts and uncertainty limits remain in the paper. Public access is not a substitute for source-data permission. The author uploaded the two archives; the agent verified them. No agent upload, git push, tag, Release, DOI, RRID, bio.tools registration or journal submission was made. Freeze an author-approved source commit and data inventory before requesting a durable archive; retain the original code snapshot used for the reported experiments.

Archive provenance note: the downloaded ZIPs preserve README/manifest statements written before the author uploaded them (for example, “local review candidate” or “no public URL”). Those are historical preparation records. The verified URLs and hashes in `metadata/external_downloads.tsv` give the current technical access status; original scientific content and ZIP bytes have not been changed.
