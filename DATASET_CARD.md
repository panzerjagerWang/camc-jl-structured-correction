# CAMC-JL metadata and numerical-results dataset v2.0.0

## Scope and provenance

759 annotation-derived cases: 200 development and 559 historical test.
The source snapshots are Natane (364 cases) and ICJS/TUFS (395 cases).
Ten supplied error categories: aspect, tense, style, passive, transitivity,
honorific, benefactive, demonstrative, sentence_connection and causative.
The underlying task is one localized correction with a supplied category.

Version 2.0.0 adds the current local-model numerical study. The older metadata
release preparation accompanied an earlier study; this version should be cited for
the six-method, three-seed comparison in the current paper.

## Files and schemas

- data/metadata.csv: release_id, split, source, error_type. All 759 rows.
- data/grouping_keys.csv: the same release IDs and opaque document_key,
  available_unit_key, window_key and pair_key labels, for overlap and cluster analysis.
  These are de-linked grouping labels, not original source IDs or text hashes.
- data/development_folds.csv: release_id, fold, component, source, error_type.
  Three grouped folds cover all 200 development rows; folds contain 67, 67 and 66 rows.
- data/test_cluster_order.csv: opaque integer cluster ranks preserving the archived bootstrap sampling order.
- data/historical_subsets.csv: pair-seen and no-shared-available-unit flags for all 559 test IDs.
- results/historical_per_example.csv: method, seed, release_id, six paper metrics,
  and two offset-based locator diagnostics. 10,062 rows; each score lies in [0,1].
- results/historical_summary.csv and historical_seed_summary.csv: archived means.
- results/historical_intervals.csv: archived paired cluster-bootstrap comparisons.
- results/development_summary.csv: development selection and control results.
- results/resources.csv: parameter and resource accounting.

The six endpoints are SpanMatch, SpanOverlapScore, RevisionMatch,
RevisionSimilarity, SoftMatch and ExactMatch. LocatorOffsetMatch and
LocatorOffsetOverlap are additional diagnostics, outside the six-endpoint family.
Scores are stored as fractions; paper tables display percentages.
Per-example scores are released numerical observations. Recomputing the original
string-based metrics requires lawful access to the excluded texts and predictions.

## Reproducible analyses

The release permits aggregate means, paired numerical comparisons, error-category
summaries and overlap-aware subgroup analysis. Three-seed paired differences are
averaged within each test example before resampling the available source units.
The published intervals use 20,000 cluster-bootstrap replicates, RNG seed 20260928,
and six-endpoint Bonferroni quantiles.

The historical data include 60 examples with a development-seen typed correction
pair, 499 without one, and 207 with no shared available source unit. There are
194 historical source-unit clusters. The latter subset is not a certified
learner-disjoint evaluation. Historical test reuse and development source overlap
limit causal or generalization claims.

## Access boundary and license

No learner writing, gold span strings, corrected strings, raw predictions,
original source IDs, private paths or provider logs are part of the dataset.
Opaque grouping labels disclose within-release overlap, not an identity mapping.
Raw examples must be obtained separately under the original providers' conditions.
This public dataset alone cannot train the correction model or independently
recompute text-level scores.

Author-created metadata and results: CC BY 4.0. See DATA_LICENSE.md.
