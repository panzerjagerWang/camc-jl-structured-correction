# CAMC-JL: structured Japanese correction

Research companion to **Structured Localization and Auxiliary Adaptation for Japanese
Grammatical Error Correction**. Project homepage: [open the site](docs/index.html).

This release package contains an **open metadata-and-numerical-results dataset**:
759 de-linked metadata records (200 development; 559 historical test) and 10,062
per-example evaluation rows (six methods × three seeds × 559 examples).
It also contains the inference method, a compact training implementation, training
recipe, figures, and checksummed model-weight download instructions.

The full learner sentences, gold spans/revisions and raw predictions are excluded
because redistribution permission has not been established. This is an evaluation
dataset, not a freely redistributable full-text training corpus.

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
retained. A fresh GPU replay of the exported auxiliary 4B package is still pending;
do not treat packaging as successful end-to-end reproduction.

Code is MIT; author-trained weights are Apache 2.0; author-created metadata,
scores and figures are CC BY 4.0. See LICENSE, LICENSE-WEIGHTS, DATA_LICENSE.md and NOTICE.
The upstream Qwen base is downloaded separately under its own Apache 2.0 license.

Publication status and the verified repository/homepage URLs are recorded in
PUBLICATION.json after the publishing script verifies the public repository and assets.
