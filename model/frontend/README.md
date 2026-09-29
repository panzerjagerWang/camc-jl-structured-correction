# Local context-gated correction frontend

This directory exports the existing crop_noposition__boundary frontend. It
contains the three original character GRUs, sparse editor/original locator,
crop-trained locator and input-only boundary gate:4,234,774 stored values.
No weights are fitted or selected during export. It is the CPU frontend of
the evaluated 4B pipeline; the 4B base and adapters are not included here.
Do not attribute the full pipeline's27.97% ExactMatch to this frontend alone.

Use a compatible Python environment with NumPy and PyTorch. The research
environments use Python3.13 for CPU audits and Python3.12 for GPU work;
exact library versions belong to the separate replay record. Reproduction is verified separately;
export by itself is not a passing replay or isolation check. No sklearn,
training corpus, project source tree or network is required for prediction.

python predict.py --list-error-types
python predict.py --text '日本語の入力文章' --error-type tense

The API is Corrector(directory,seed).predict(text,category). The fixed default
seed20260927 was chosen before historical scoring. All three seeds are
included. A supplied category and one local correction are required; this
is not open error detection or a verified educational intervention. The
inspected external localization failure is retained. This local export
does not imply public deposition or unrestricted corpus licensing.
