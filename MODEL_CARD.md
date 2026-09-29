# Selected structured correction model

## Intended use

Research on category-conditioned, single-span Japanese correction. The caller
provides Japanese text and one supported error category. The system localizes an
interval, proposes a compact edit, and conditionally replaces that edit with a
4B decoder output. It is not open error detection or a validated teaching tool.

## Architecture and released weights

- Context-gated frontend: 4,234,774 stored values, comprising three seeded GRUs,
  sparse edit/original-localization components, a crop locator and a boundary gate.
- Base: Qwen/Qwen3-4B-Instruct-2507, pinned revision
  cdbee75f17c01a7cc42f958dc650907174af0554, 4,022,468,096 parameters.
- LoRA: 16,515,072 parameters per seeded adapter, rank 8, alpha 16, dropout 0.05,
  on q/k/v/o and gate/up/down projections.
- Full inference pipeline: 4,043,217,942 parameters/coefficients.
- Three final adapters: seeds 20260927, 20260928, 20260929. Default seed 20260927
  was fixed before historical scoring. Base is BF16; adapters are float32.

The eight release assets contain five frontend weight files and three adapters.
WEIGHTS.json maps names to runtime paths and audited SHA-256 digests.
model/adapters/*/adapter_config.json replaces an original machine path with the
public base model ID and pinned revision; the adapter tensors are unchanged.
Base files are fetched separately from Hugging Face at the pinned revision.

## Training

16,000 distinct auxiliary examples (1,389 teacher-authored simulated pairs and
14,611 automatic pairs) were sampled from a 59,767-example eligible pool.
Auxiliary adaptation uses 1,000 updates, effective batch 16, microbatch 2 and
peak learning rate 2e-4. Each final adapter then fits 200 source-development
examples for 10 epochs / 130 updates, with a fresh AdamW optimizer, peak 1e-4,
and the original 390-update schedule horizon. All three refits use the same
auxiliary step-1000 adapter as initialization. See method/README.md.

Neither auxiliary training rows nor protected source text are included.
The public training CLI exposes the fitting algorithm for user-supplied lawful
inputs; it is not a replacement for the original data provenance and selection audit.

## Decoding

Four-beam deterministic generation uses length penalty 0.6. A generated edit
must be complete, nonempty, changed, single-line and within the length bound.
It is accepted only when its EOS-inclusive length-normalized log likelihood
strictly exceeds that of the compact edit. Otherwise the compact edit is retained.
The selected interval is unchanged. See model/predict.py.

## Evidence and limitations

Historical ExactMatch is 33.04%, averaged across three seeds on 559 examples.
The matched 4B source-only comparator reaches 27.97%. Localization does not improve.
An archived hosted comparison reaches 40.61%, but complete outputs are unavailable
for an independent paired rescore. Global state of the art is not established.
Historical test reuse, overlap, and poor external localization limit transfer claims.

The compact frontend has an existing isolated replay audit. The full auxiliary
portable export is marked exported_pending_verification: a fresh end-to-end
GPU replay of this export has not passed in the current packaging session.
Audited source hashes accompany unchanged tensor assets; publisher/downloader
scripts independently verify asset hashes before publication/use.

The upstream shard-index total exceeds the actual published tensor-file total
by 655,360 bytes. The study retained the unchanged publisher files after a
separate architecture and file-hash audit; this is a disclosed publisher metadata
discrepancy, not an altered weight shard.

Author-trained weights use Apache 2.0. The upstream base's license and notices
remain applicable. The frontend includes learned character/edit vocabularies,
but no training-example table or private source-identity mapping.
