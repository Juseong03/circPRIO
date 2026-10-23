# Prepared data

Included for `circcnns`, `circ2lo`, `k562`, `GM12878` and `HepG2`:

```text
data/datasets/<dataset>/candidates.parquet
data/windows/<dataset>_L400.npz
data/windows/<dataset>_L400.json
```

Candidate columns are `candidate_id`, `chromosome`, `start0`, `end0`, `strand`, `y`, `component`, `fold`. Labels are 0/1; fixed folds are 0–4 for training populations and -1 for frozen transfer. Candidate order and component-controlled splits are preserved. File sizes/SHA-256 values are in `metadata/github_data_files.tsv`.

Window NPZs contain `windows`: uint8 `(N,2,400)`, acceptor then donor, with A=0, C=1, G=2, T=3 and invalid=255. JSON manifests record row counts, invalid candidates and the raw-array SHA-256. Windows are genomic and transcript-oriented. The embedding routine uses placeholder bases for invalid rows; when excluding them for custom inference, preserve the mapping to candidate IDs.

## External RCM arrays

Download/extraction/checksum commands are in the main [README](../README.md#data-and-downloads). Extract the RCM ZIP into the repository root to create `data/datasets/<dataset>/rcm.npy` and `data/rcm_SHA256SUMS`.

RCM arrays are float32 `(N,3,5,250)`, containing log-transformed CircCNNs descriptors aligned to candidate-table order. They are input descriptors rather than model weights. Do not standardise them for TriRCM. Per-array sizes/hashes are in `metadata/rcm_files.tsv`.

To regenerate descriptors, see [Regenerating RCM](../README.md#regenerating-rcm). The benchmark populations use fixed 3,000-nt flanks; cell-line populations cap each flank at the adjacent annotated intron. The committed `data/datasets/{k562,GM12878,HepG2}/rcm_flanks.parquet` tables give `candidate_id`, `upstream_intron_nt` and `downstream_intron_nt` in candidate order.

SpliceBERT weights are downloaded separately, embeddings are generated locally, and trained circPRIO checkpoints are obtained separately or trained with the provided scripts. The external reproducibility ZIP contains selected K562 candidate-level predictions and figure sources; it does not contain a complete trained-model collection. See the main [reproduction instructions](../README.md#reproducing-the-reported-results).
