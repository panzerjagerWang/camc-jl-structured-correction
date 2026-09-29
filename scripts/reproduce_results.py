"""Reproduce means and matched 4B paired intervals from released numerical data."""
from pathlib import Path
import csv, json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["SpanMatch", "SpanOverlapScore", "RevisionMatch",
           "RevisionSimilarity", "SoftMatch", "ExactMatch"]
SEEDS = ["20260927", "20260928", "20260929"]
PRIMARY = "qwen4b_aux16k_selected_primary"
MATCHED = "qwen4b_selected_primary"

def read(name):
    with (ROOT / name).open(newline="") as f:
        return list(csv.DictReader(f))

def bootstrap_bounds(delta, labels):
    labels = np.asarray(labels)
    unique = sorted(set(labels))
    sums = np.asarray([delta[labels == key].sum(0) for key in unique])
    counts = np.asarray([(labels == key).sum() for key in unique])
    rng = np.random.default_rng(20260928)
    samples = []
    for first in range(0, 20000, 256):
        draw = rng.integers(len(counts), size=(min(256, 20000-first), len(counts)))
        samples.append(sums[draw].sum(1) / counts[draw].sum(1)[:, None])
    return np.quantile(np.concatenate(samples) * 100,
                       [.025, .975, .05/12, 1-.05/12], axis=0)

def main():
    metadata = read("data/metadata.csv")
    rows = read("results/historical_per_example.csv")
    groups = {r["release_id"]: r for r in read("data/grouping_keys.csv")}
    assert len(metadata) == 759 and len(groups) == 759 and len(rows) == 10062
    methods = sorted({r["method"] for r in rows})
    ids = sorted(r["release_id"] for r in metadata if r["split"] == "historical_test")
    assert len(methods) == 6 and len(ids) == 559
    lookup = {(r["method"], r["seed"], r["release_id"]): r for r in rows}
    assert len(lookup) == len(rows)
    arrays = {}
    for method in methods:
        arrays[method] = np.asarray([[[float(lookup[method, seed, i][m])
                                      for m in METRICS] for i in ids] for seed in SEEDS])
        assert np.isfinite(arrays[method]).all()
        assert ((arrays[method] >= 0) & (arrays[method] <= 1)).all()
    archive = {r["method"]: r for r in read("results/historical_summary.csv") if r["subset"] == "all"}
    means = {}
    for method, a in arrays.items():
        mean = a.mean(axis=(0, 1))
        np.testing.assert_allclose(mean, [float(archive[method][m]) for m in METRICS], atol=1e-12, rtol=0)
        means[method] = dict(zip(METRICS, (mean*100).tolist()))
    delta = (arrays[PRIMARY]-arrays[MATCHED]).mean(axis=0)
    cluster_order = {r["release_id"]: int(r["available_unit_rank"]) for r in read("data/test_cluster_order.csv")}
    labels = [cluster_order[i] for i in ids]
    assert len(set(labels)) == 194
    bounds = bootstrap_bounds(delta, labels)
    archived_intervals = {r["metric"]: r for r in read("results/historical_intervals.csv")
                          if r["subset"] == "all" and r["baseline"] == MATCHED and r["family"] == "legacy_six"}
    np.testing.assert_allclose(bounds, [[float(archived_intervals[m][key]) for m in METRICS]
        for key in ["ci95_lower_pp", "ci95_upper_pp", "simultaneous95_lower_pp", "simultaneous95_upper_pp"]], atol=1e-10, rtol=0)
    result = {
        "historical_means_percent": means,
        "paired_primary_minus_matched_percentage_points": dict(zip(METRICS, (delta.mean(0)*100).tolist())),
        "paired_bounds": {m: dict(zip(["pointwise_low", "pointwise_high", "six_metric_low", "six_metric_high"],
                                     bounds[:, j].tolist())) for j, m in enumerate(METRICS)},
        "clusters": 194, "bootstrap_replicates": 20000,
        "note": "Recomputes numerical analyses; excluded raw text is needed to rescore predictions."
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
