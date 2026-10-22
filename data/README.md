# Prepared data

## Included in GitHub

The following real files are committed with the code, for `circcnns`, `circ2lo`, `k562`, `GM12878` and `HepG2`:

```text
data/datasets/<dataset>/candidates.parquet
data/windows/<dataset>_L400.npz
data/windows/<dataset>_L400.json
```

Candidate tables contain `candidate_id`, `chromosome`, `start0`, `end0`, `strand`, `y`, `component`, `fold`. The label is 0/1; fixed folds are 0–4 for training populations and -1 for transfer populations. Row order and component-controlled folds are preserved. `metadata/github_data_files.tsv` records file sizes and SHA-256 checksums.

Compressed window archives contain one `windows` array: uint8 `(N, 2, 400)`, acceptor then donor, encoded A=0, C=1, G=2, T=3, invalid=255. The JSON manifest records the dataset, row count and SHA-256 of the uncompressed raw array bytes. Compression preserves all values. The loader reads these files directly; no manual unpacking is needed.

Windows are genomic and transcript-oriented, not spliced transcript sequence. Invalid rows are retained and listed in the manifests. Exclude windows containing 255 when preparing inference features; the embedding routine uses placeholder bases for these rows. Preserve the exclusion-to-candidate-ID mapping.

## RCM data: Google Drive

**Download `circPRIO_RCM_20261021.zip`: [Google Drive link to be added by the repository owner].**

This archive is the only dataset download needed for circPRIO-RCM. It is unnecessary for circPRIO. Extract it into the repository root to create:

```text
data/datasets/<dataset>/rcm.npy
data/rcm_SHA256SUMS
```

RCM arrays are float32 `(N, 3, 5, 250)`, containing the log-transformed CircCNNs descriptors in candidate-table order. They are input descriptors, not model weights or learned embedding caches. CircCNNs native key/index alignment was verified before distribution. Do not standardise the descriptor for the adopted TriRCM model.

Verify the external files after extraction:

```bash
sha256sum -c data/rcm_SHA256SUMS
```

`metadata/rcm_files.tsv` records the external files' sizes and hashes. These large arrays are ignored by Git. No Drive upload has been performed by this packaging process.

### Recomputing RCM from sequence

`scripts/make_rcm.py --dataset <dataset> --genome <fasta>` computes the same arrays from an indexed reference genome (hg19: `circcnns`, `circ2lo`; hg38: `k562`, `GM12878`, `HepG2`). Flanks are 3,000 nt in transcript orientation outside the back-splice ends. For `k562`, `GM12878` and `HepG2` each flank is capped at the adjacent annotated intron; `data/datasets/<dataset>/rcm_flanks.parquet` (`candidate_id`, `upstream_intron_nt`, `downstream_intron_nt`, candidate order) gives these lengths. The recomputed arrays are identical to the distributed ones for every candidate.

## Generated and optional assets

SpliceBERT model weights are downloaded using `scripts/download_splicebert.py`. Regenerate frozen embeddings from the included compressed windows with `scripts/embed.py`; generated caches remain under `artifacts/` and are ignored by Git.

Trained circPRIO weights are not included in GitHub. Training exports complete checkpoints and companion configs; existing checkpoints can be distributed separately. The original all-data and checkpoint ZIPs remain available in the working project's `release/downloads/` directory, but the all-data ZIP is unnecessary for this repository layout.
