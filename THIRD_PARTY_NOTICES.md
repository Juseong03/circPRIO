# Attribution and licensing

circPRIO-RCM uses the reverse-complementary-match descriptor and TriRCM encoder from **CircCNNs** (Wang et al.; original RCM_optuna_concate_10000). The local TriRCM implementation is part of circPRIO-RCM; this attribution does not claim a new RCM architecture. `src/circprio/rcm_generate.py` is a reimplementation of the CircCNNs RCM descriptor computation (Auxiliary_Codes/RCM_features_generation.py) that reproduces the CircCNNs descriptor exactly. Comparator implementations are not included in this repository.

Frozen sequence states come from `multimolecule/splicebert`, revision `8e3f4e5088b3bbf76876d8b931078d9f03caffdd`. Its weights are downloaded separately and retain their upstream terms.

No new license for the authors' code or derived datasets is assigned by this packaging step. Repository owners must choose applicable distribution terms before public release. This package does not grant additional rights over third-party material.
