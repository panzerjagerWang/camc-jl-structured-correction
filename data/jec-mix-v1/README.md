# JEC-Mix 1.0 — public JWTD training component

**JEC-Mix: Japanese Error Correction Mixed-source Corpus** combines the preserved
learner benchmark with auxiliary correction sources. We open-source its
**50,000-record JWTD-derived full-text training component** under **CC BY-SA 3.0**,
with source sentences, corrections, automatic spans and original revision
provenance. [Download JSONL gzip](https://github.com/panzerjagerWang/camc-jl-structured-correction/releases/download/v2.1.0/JEC-Mix-v1.0-JWTD-train-50000.jsonl.gz) · [manifest](manifest.json) ·
[attribution](ATTRIBUTION.md) · [license](LICENSE-CC-BY-SA-3.0.txt).

This component is a derived subset of the [official Kyoto University JWTD v2
training partition](https://nlp.ist.i.kyoto-u.ac.jp/index.php?%E6%97%A5%E6%9C%AC%E8%AA%9EWikipedia%E5%85%A5%E5%8A%9B%E8%AA%A4%E3%82%8A%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88), not an official new release by its publishers.
It contains 40,439 article groups. Publisher `test`/`gold` contents were not
extracted or parsed, and are not included.

## Mixed collection and release scope

| Source / role | Rows | Full text public here? |
|---|---:|---|
| Learner core: development | 200 | No; existing de-linked metadata only |
| Learner core: historical evaluation | 559 | No; never auxiliary training |
| Teacher-generated auxiliary | 5,069 | No |
| Automatic synthetic auxiliary | 54,674 | No |
| JWTD Wikipedia-revision auxiliary | 50,000 | **Yes, CC BY-SA 3.0** |
| Auxiliary training total | 109,743 | Only the JWTD component |
| Core and auxiliary total | **110,502** | Whole mix is **not fully open** |

These totals describe different sources and task roles; they are not 110,502
human learner annotations. Rights to Natane/ICJS learner texts remain unchanged.
The existing CAMC-JL v2.0.0 release remains an open metadata-and-results dataset.
No private learner/teacher/synthetic rows, protected forms, evaluation requests
or raw model predictions are included in this release.

## Conversion and selection

Source and corrected text and all publisher metadata remain literal. Exactly
one native diff must have exactly one placement that reconstructs the full
corrected sentence. A minimal changed hull supplies zero-based, half-open Unicode
code-point offsets. Pure insertion/deletion is anchored to an unchanged character
on the right, otherwise the left, so source span and replacement are nonempty.
Inputs have 4–256 characters; spans 1–40; replacements at most `max(44,4*span_length)`.
Reserved prompt/chat markers, multiline inputs, unchanged pairs, publisher
`not-typo` and whitespace-edged replacements are excluded.

NFKC/Unicode-whitespace removal is used only for identity and screening, never to
rewrite source text. Before filtering or sampling, a joint graph includes all
696,189 publisher training rows, their article IDs and all old auxiliary pairs;
discarded/unsampled rows remain graph bridges. Screening against explicitly
supplied protected forms excluded 15 components, 14 publisher rows and 24 teacher
rows. It does not certify every semantic or pretraining overlap. The protected
texts and old auxiliary graph edges cannot be publicly redistributed. Their
full graph cannot be independently rebuilt from this public component alone.

Algorithm R, seed 20261009, selected 50,000 records from 621,915 eligible unique
pairs in pinned publisher line order. Order affects reservoir selection. Native
labels are preserved separately; `error_type=external_typo` is an auxiliary domain,
not a mapping into the ten learner categories. Fields named `gold_span` and
`gold_revision` express training targets in the existing schema: these spans are
**automatically derived**, not human minimal-span annotations. Publisher
likelihoods are provenance, not evaluation scores from our model.

## Inspect and verify

The artifact contains newline-delimited UTF-8 JSON. Only Python 3.11+ is needed:

```bash
python scripts/verify_jec_mix_jwtd.py JEC-Mix-v1.0-JWTD-train-50000.jsonl.gz
gzip -dc JEC-Mix-v1.0-JWTD-train-50000.jsonl.gz > JEC-Mix-v1.0-JWTD-train-50000.jsonl
# Optional: independently download/extract the official train member only.
python scripts/verify_jec_mix_jwtd.py JEC-Mix-v1.0-JWTD-train-50000.jsonl.gz --publisher-train train.jsonl
```

The verifier checks compressed/uncompressed SHA-256, 50,000 unique records,
40,439 article groups, full literal reconstruction, normalized pair hashes,
native metadata and every per-row conversion. With `--publisher-train`, it replays
every retained record against the original publisher line and line hash.
It does not load models, score corrections or read publisher test/gold.
`scripts/jec_mix_jwtd_conversion.py` publishes the unchanged per-row transformation
math from the pinned producer. It does not expose the private quarantine inputs.

Uncompressed SHA-256: `e30fc2cd36c15bad6fd35373e1a413fd4a84b0712af9473ad308fad7f32cc8f8` (73,901,779 bytes).
Compressed SHA-256: `d4dc00fa77f4eaff4cd0c38ca4cea25c3a009a83ed279b21a646117ee94911f5` (20,878,393 bytes).
All SHA values are locally measured; the publisher supplied no corpus checksum.

## Model status and limits

Existing published v2.0.0 model weights predate this JWTD subset. They contain
no training on it. A later matched development study trained 1,319,003-parameter
character locators with and without JWTD at matched budgets. The Wikipedia arm
was weaker than the NoWiki control. No new weights accompany this dataset release,
and no state-of-the-art claim follows from corpus size or reused-development
experiments. The new mixed collection is available for separately frozen,
source-aware training and controlled evaluation.

[Attribution and source revision links](ATTRIBUTION.md) are part of this release.
Use its CC BY-SA 3.0 terms for this component; do not apply the existing CC BY 4.0
metadata license or model-weight license to these Wikipedia texts.
