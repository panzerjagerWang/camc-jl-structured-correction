# CAMC-JL: structured Japanese correction

Research companion to **SLA-JEC: Structured Localization and Auxiliary Adaptation for Japanese
Grammatical Error Correction**. Project homepage: [open the site](https://panzerjagerwang.github.io/camc-jl-structured-correction/).

**[Paper homepage](https://panzerjagerwang.github.io/camc-jl-structured-correction/)** ·
**[Latest manuscript PDF](https://panzerjagerwang.github.io/camc-jl-structured-correction/paper/manuscript.pdf)** ·
**[Metadata/results ZIP](https://github.com/panzerjagerWang/camc-jl-structured-correction/releases/download/v2.0.0/CAMC-JL-metadata-results-v2.0.0.zip)** ·
**[Model weights](https://github.com/panzerjagerWang/camc-jl-structured-correction/releases/tag/v2.0.0)**

The manuscript and pipeline figures reflect the author's revision of 2 October 2026.
[Editable manuscript sources](docs/downloads/SLA-JEC-manuscript-source-20261002.zip)
include LaTeX, bibliography and vector figures. The historical metadata/results dataset and numerical
observations remain v2.0.0. The separate JEC-Mix 1.0 JWTD training component is
released in repository version v2.1.0.

This release package contains an **open metadata-and-numerical-results dataset**:
759 de-linked metadata records (200 development; 559 historical test) and 10,062
per-example evaluation rows (six methods × three seeds × 559 examples).
It also contains the inference method, a compact training implementation, training
recipe, figures, and checksummed model-weight download instructions.

The full learner sentences, gold spans/revisions and raw predictions are excluded
because redistribution permission has not been established. This is an evaluation
dataset, not a freely redistributable full-text training corpus.

## JEC-Mix 1.0: open JWTD training component

We open-source **50,000 full-text Japanese correction pairs** derived from the
**official JWTD v2 training partition**, under **CC BY-SA 3.0**. The component
retains literal source/corrected sentences, automatic spans, native categories,
article/revision metadata and source-line hashes across 40,439 article groups.
[Download JSONL gzip](https://github.com/panzerjagerWang/camc-jl-structured-correction/releases/download/v2.1.0/JEC-Mix-v1.0-JWTD-train-50000.jsonl.gz) ·
[Dataset card](data/jec-mix-v1/README.md) · [Attribution](data/jec-mix-v1/ATTRIBUTION.md) ·
[Manifest/checksums](data/jec-mix-v1/manifest.json).

**JEC-Mix: Japanese Error Correction Mixed-source Corpus** contains 110,502
core and auxiliary records: 200 development + 559 historical learner records,
5,069 teacher-generated, 54,674 synthetic and 50,000 JWTD rows. These are distinct
sources and task roles, not 110,502 human learner annotations. The whole mix is
**not fully open**; only the JWTD full-text component is released here. Raw
Natane/ICJS learner texts, teacher/synthetic rows, protected forms and evaluation
requests/predictions are excluded. Publisher test/gold are not included or parsed.

Existing published model weights predate JWTD. A later matched locator study
found the Wikipedia arm weaker than its NoWiki control. Dataset size does not
establish SOTA, and this dataset release provides no newly trained weights.
The per-row transformation math and a no-dependency verifier are public:

```bash
python scripts/verify_jec_mix_jwtd.py JEC-Mix-v1.0-JWTD-train-50000.jsonl.gz
# Optionally check every original line in the official training member:
python scripts/verify_jec_mix_jwtd.py JEC-Mix-v1.0-JWTD-train-50000.jsonl.gz --publisher-train train.jsonl
```

## Contents

- [Dataset card](DATASET_CARD.md), [metadata CSV](data/metadata.csv) and [numerical results](results/).
- [Model card](MODEL_CARD.md) and [training/decoding method](method/README.md).
- [Recompute published means and paired intervals](scripts/reproduce_results.py).
- [Pipeline figure](docs/assets/Fig1_pipeline.pdf) and [editable Visio](docs/assets/Fig1_pipeline_editable.vsdx).
- [Manuscript draft](docs/paper/manuscript.pdf).

## Install and run

Use Python 3.12 for the pinned 4B environment. Full decoding requires a compatible
CUDA GPU; the compact frontend runs on CPU. The historical GPU was a 16 GB
RTX 4070 Ti SUPER. The 4B pipeline has 4.043 billion stored parameters/coefficients;
the classroom-scale CPU frontend has 4.235 million.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r model/requirements.txt
# From a Git clone, the repository is inferred from origin.
python scripts/download_weights.py --frontend-only
python model/frontend/predict.py --list-error-types
python model/frontend/predict.py --text '日本語の入力文章' --error-type tense
# Download the three adapters and the pinned upstream base (~8 GB) as well:
python scripts/download_weights.py
python model/predict.py --text '日本語の入力文章' --error-type tense
python scripts/reproduce_results.py
```

For a downloaded source ZIP, pass `--repo OWNER/REPOSITORY` using the actual release
repository shown in your browser. The downloader validates SHA-256 hashes before
moving each downloaded file into place. No API token is required for public assets.
Frontend-only installation can use `pip install numpy==2.3.1 torch==2.11.0`.

## Historical results

| Method | ExactMatch (%) |
|---|---:|
| Character GRU | 14.31 |
| Productive fusion | 17.05 |
| 1.7B auxiliary/source | 26.77 |
| 1.7B context frontend | 26.71 |
| Matched 4B source-only | 27.97 |
| 4B auxiliary + source (selected) | **33.04** |

Values are means over the same 559 historical examples and three seeds.
The selected method improves ExactMatch over matched 4B source-only by 5.07 percentage
points; the six-endpoint adjusted paired cluster interval is [1.18, 9.32].
Localization ties the matched 4B comparator. Incremental RevisionSimilarity and
SoftMatch intervals include zero. No global state-of-the-art claim is made.

The historical test set was reused during development and has source overlap.
External localization exposed substantial transfer failure. These scores measure
retrospective annotation agreement, not classroom effectiveness.

## Verification and licenses

Numerical data structure, release IDs, complete method/seed coverage and historical
means were checked during packaging. Existing compact-frontend verification is
retained. The existing exported auxiliary 4B package was validated locally on GPU on
9 October 2026. This dataset release adds no model weights.

Code is MIT; author-trained weights are Apache 2.0; author-created metadata,
scores and figures are CC BY 4.0. See LICENSE, LICENSE-WEIGHTS, DATA_LICENSE.md and NOTICE.
The upstream Qwen base is downloaded separately under its own Apache 2.0 license.

Publication status and the verified repository/homepage URLs are recorded in
PUBLICATION.json after the publishing script verifies the public repository and assets.
