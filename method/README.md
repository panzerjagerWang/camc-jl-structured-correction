# Method and reproduction boundaries

The released inference implementation is the existing portable export, not a
new approximation. model/frontend/ contains the GRU, sparse edit and localization
components. model/predict.py implements the frozen-interval 4B generation and strict
likelihood fallback. The current figure depicts the same steps.

## Fitting recipe

method/train_adapter.py is a compact extraction of the original prompt encoding,
4B loader, response-token-weighted microbatch update and two fitting schedules.
It removes dependencies on private audit files. It accepts externally supplied
JSONL; it does not download, redistribute or synthesize restricted training rows.

Each input object needs:
`text`, `error_type`, `gold_span_start`, `gold_span_end`, `gold_revision`.
Offsets index Unicode characters in the unnormalized input text, with an exclusive
end. Prompt labels are masked; only replacement tokens and EOS are supervised.
All encoded rows must fit within 512 tokens. It does not silently truncate or
drop rows.

For the historical auxiliary recipe, input order must be the original 5,093
teacher rows followed by 54,674 automatic rows. The fixed sampler chooses
16,000 distinct positions using NumPy default_rng(20260928). Without the same
lawfully obtained data and original row order, this is a new fit, not reproduction
of the published checkpoint.

```bash
python method/train_adapter.py --stage auxiliary --input /path/to/authorized_aux.jsonl --output /path/to/aux_fit --seed 20260928
python method/train_adapter.py --stage source --input /path/to/authorized_dev.jsonl --initial-adapter /path/to/aux_fit --output /path/to/source_fit --seed 20260927
```

Repeat the source stage from the SAME auxiliary output for seeds 20260928 and
20260929. Never continue fitting from another source seed or select a seed using
historical outcomes. The auxiliary stage uses 1,000 updates at effective batch 16.
Source fitting uses 200 rows × 10 epochs, with 130 updates and a fresh optimizer.
Its learning-rate horizon remains 390 updates, not 130. The code preserves the
tail batch of eight examples. Both stages use microbatch two, AdamW weight decay
0.01, gradient clipping 1.0, 5% warmup and cosine decay to a 0.1 multiplier.

The original grouped development protocol, candidate selection, stress transformations,
and front-end fitting are described in the paper. This CLI fits adapters; it does
not re-run that selection study or train every frontend baseline. The fitted frontend
is supplied directly. The exported auxiliary starting adapter is not a release asset;
the CLI can recreate a new auxiliary fit from authorized inputs. The original run
initialized from a saved untrained adapter; bitwise identity of a new initialization
and the historical adapter is not asserted.

## Scoring and numerical analysis

Six paper endpoints are precomputed and released in results/. The legacy substring
scorer resolves repeated occurrences using maximum gold overlap and nearest gold
position. Offset diagnostics are also retained. This behavior can make localization
agreement optimistic. Numerical-only reproduction cannot audit excluded strings.

`python scripts/reproduce_results.py` recalculates six-metric means and the paired
primary-versus-matched 4B intervals. It averages three-seed differences per example
and resamples 194 available source-unit clusters 20,000 times. The six-endpoint
family uses Bonferroni tail probabilities; it does not correct the entire adaptive
research history.

## Validation status

Public inference files and tensors are copied from the archived export. The
standalone fitting wrapper and downloader were inspected during packaging, but
Python subprocesses could not launch in that session. No new training, external
benchmark or successful auxiliary GPU replay is claimed. This release preserves
that limitation explicitly.
