# Changelog

All notable changes to ACWS's AI pipeline and serving backend are recorded here.
Dates are in `YYYY-MM-DD`.

## 2026-09-17

### Reproducibility fix

- `AI/outputs/dataset_registry.json` and `AI/outputs/metrics.json` previously showed
  zero datasets found and metrics from an unrelated machine
  (`/home/user/dev/aiot/...`). Those artifacts were never reproducible from this
  repository and have been treated as untrustworthy throughout this work.
- `AI/outputs/*.json`, `AI/outputs/*.pt`, `AI/outputs/splits/`, and `AI/outputs/eval/`
  are now gitignored (`AI/outputs/.gitignore`); every artifact referenced from the
  README is regenerated locally from `AI/dataset/` and carries an
  `artifact_manifest.json` fingerprint tying it to the dataset registry that produced
  it.

### Heuristic removal

- `backend/app/services/model_service.py`'s `_tabular_adapter_probability` heuristic,
  which silently blended the trained model's output with a hardcoded formula
  whenever confidence was near 0.5, has been removed.
  `predict_positive_probabilities` now returns the raw model probability only.
  `tests/test_model_service.py::test_heuristic_tabular_adapter_probability_removed`
  is a regression guard against reintroducing it.

### Per-dataset graphs, real embeddings, and honest metrics

- `AI/main.py::run_pipeline_for_dataset` trains one isolated graph per dataset
  (FakeNewsNet, LIAR, PHEME) instead of concatenating them into a single merged
  graph. LIAR's graph is intentionally edge-free (its source statements carry no
  user/news relationships); this is disclosed as a controlled structural
  difference between datasets, not hidden inside a merged graph.
- **Found and fixed:** `sentence-transformers` was listed in `AI/requirements.txt`
  but was not actually importable in the pipeline's virtual environment. Every
  prior per-dataset run had silently fallen back to hashing text features despite
  the code defaulting to `embedding_backend="sentence-transformer"`. Installed the
  dependency (pinned to `sentence-transformers==3.0.1` /
  `transformers==4.41.2` for compatibility with the installed CUDA-enabled torch
  build) and reran all three datasets with real semantic embeddings.
- **Found and fixed:** training was capped at 6 epochs with early-stopping patience
  2. On FakeNewsNet this was too little to escape a majority-class collapse — the
  model predicted "positive" for all 46 test examples. Raised the defaults to 60
  epochs / patience 10 (`AI/main.py`, `AI/src/gnn_pipeline/gcn_baseline.py`); both
  are still overridable via CLI flags.
- **Found and fixed:** TorchScript export (`torch.jit.script`) failed for every GAT
  run because PyG's `GATConv` cannot be scripted on the installed torch_geometric
  version (`Could not cast value of type Optional[Tensor] to bool`). Training
  silently fell back to state-dict-only export. Added a `torch.jit.trace` fallback
  in `AI/src/gnn_pipeline/train.py`; exports now succeed for the GAT (via trace)
  and the GCN baseline (via script). The export method is recorded in
  `metrics.json["artifacts"]["torchscript_export_method"]`.
- Added a `--mode per-dataset --dataset-name {fakenewsnet,liar,pheme}` CLI entry
  point to `AI/main.py` so each dataset can be (re)trained with one command instead
  of an ad hoc script.
- Extended `AI/src/gnn_pipeline/evaluate.py` with `compute_extended_metrics`
  (ROC-AUC, PR-AUC) and added `AI/src/gnn_pipeline/report_eval.py`, which
  re-evaluates the frozen, persisted test split for a dataset and writes
  `eval/eval_metrics.json` plus confusion-matrix/ROC/PR plots.
- Added `AI/src/gnn_pipeline/macro_average.py` (unweighted mean across datasets —
  deliberately not weighted by size, since the question being asked is "how well
  does this architecture generalize across structurally different domains," not
  "aggregate raw performance").

### Baselines

- Added `AI/src/gnn_pipeline/gcn_baseline.py` (four-layer residual GCN, same
  per-dataset splits and training loop as the GAT) and
  `AI/src/gnn_pipeline/baseline_macro_average.py` (macro-averages the GCN and the
  per-dataset logistic-regression surrogate that `train_and_evaluate` already fits
  alongside every GNN run).

### Ablation

- Added `AI/src/gnn_pipeline/ablation.py`: leave-one-out ablation over
  `[velocity, bot_ratio, echo_density, depth]` against the live FakeNewsNet serving
  model, cross-checked against mean absolute SHAP importance from
  `xai_service.explain()` on the same test examples.

### Backend fixes found while wiring this up

- `backend/requirements.txt` was missing `torch` and `torch-geometric`, even
  though `model_service.py` hard-imports both to load the GNN checkpoint — the
  backend could not actually be installed from its own requirements file. Added
  both.
- `backend/app/core/config.py`'s `BACKEND_MODEL_PATH` pointed at a stale
  `backend/model/model.pkl` left over from the old combined-graph run, not the
  per-dataset logistic-regression surrogate that actually backs the configured
  serving dataset. It now derives from a single `DEFAULT_SERVING_DATASET`
  constant (`"fakenewsnet"`), which also drives `AI_OUTPUTS_DIR`, so the artifact
  path and the model path can't drift apart again. Removed the unused
  `BACKEND_MODEL_METADATA_PATH` constant (declared, never referenced).
- `AI/src/gnn_pipeline/loaders.py` referenced `sparse.spmatrix` and
  `"scipy.sparse.spmatrix"` in type annotations without importing `scipy.sparse`
  as `sparse` — a latent `NameError` risk masked only by `from __future__ import
  annotations` deferring evaluation. Fixed the import.
- Regenerated `requirements-lock.txt` (pip freeze of `AI/.venv`) to match the
  updated `transformers`/`sentence-transformers`/`tokenizers`/`huggingface_hub`
  pins.

### Tests and CI

- Added `tests/` at the repo root: `test_simulation_service.py`,
  `test_propagation_service.py`, `test_model_service.py` (includes the heuristic
  regression guard), `test_api.py` (FastAPI `TestClient` against `/`, `/simulate`,
  `/predict`, `/explain`), and `test_pipeline_per_dataset.py` (runs
  `run_pipeline_for_dataset` end-to-end against a tiny synthetic FakeNewsNet-style
  fixture — no real dataset or network access required).
- Added `.github/workflows/ci.yml` with three jobs: `lint` (ruff, restricted to
  syntax-error / undefined-name rules — see `ruff.toml`), `backend-tests`
  (installs `backend/requirements.txt`, runs the FastAPI-facing tests), and
  `ai-pipeline-tests` (installs `AI/requirements.txt`, runs the pipeline fixture
  test).

## Results

See the README "Results" section for the current per-dataset and macro-averaged
numbers. They are regenerated from this session's runs, not the earlier degenerate
(majority-class-collapsed) ones.

## Known limitations carried forward

- `risk_level` in `explanation_engine.py` is rule-based, not model-derived
  (unchanged from the original review — no phase in this pass targeted it).
- The runtime 4-feature vector (`velocity`, `bot_ratio`, `echo_density`, `depth`) is
  simulator-derived online but dataset-derived offline; they are different
  data-generating processes and are not directly comparable, as already documented
  in the README.
- `AI/src/gnn_pipeline/model.py` and `backend/app/services/model_service.py` each
  define their own copy of `GraphGATClassifier` / `ResidualGATBlock`. They must be
  kept in sync by hand; nothing currently enforces that.
- LIAR's loader (`AI/src/gnn_pipeline/loaders.py::load_liar`) extracts only `id`,
  `statement`, and `label` from the TSVs. It discards the speaker-history columns
  (`barely_true`, `false_count`, `half_true_count`, `mostly_true_count`,
  `pants_fire_count`) and `party`/`subject`/`speaker` metadata, which are known in
  the LIAR literature to carry more signal than the statement text alone. Adding
  them as auxiliary node features is a plausible way to move LIAR off its
  near-majority-class baseline, but was out of scope for this pass (it changes the
  per-dataset feature schema, not just the training loop).
- `AI/config.yaml` (seed, split ratios) is not read by any code; the same values
  are hardcoded directly in `AI/src/gnn_pipeline/train.py`. Either wire it up or
  remove it.
