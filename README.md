# AI-Driven Cognitive Warfare Simulator (ACWS)

Full-stack misinformation analysis platform that combines narrative generation, propagation simulation, model-based detection, and explainable AI reporting.

## Overview

Misinformation response tooling often separates content generation, diffusion analysis, and model explainability into different systems. ACWS integrates these stages into a single workflow:
- Generate or ingest narrative content
- Simulate how it propagates through a social graph
- Classify risk using trained artifacts
- Explain why the system produced that classification

This matters in real-world moderation, fact-checking triage, and policy analysis where decisions must be both fast and auditable. ACWS differs from typical text-only classifiers by modeling propagation behavior explicitly (velocity, network structure, bot participation, cascade depth) and coupling output with SHAP-driven explanations.

## System Capabilities

### Generation
- Generates misinformation-style narratives from a topic (`/generate/misinformation`)
- Generates counter-narratives (`/generate/counter`)
- Generates neutral reports (`/generate/neutral`)
- Supports local fallback generation when Ollama is unavailable (configurable)

### Simulation
- Simulates spread on a synthetic network generated from input text
- Computes propagation metrics: reach, depth, velocity, echo chamber density, per-step spread
- Supports runtime control overrides:
  - `bot_ratio_override`
  - `seed_nodes`
  - `influencer_strength`

### Detection
- Predicts misinformation risk from feature vector `[velocity, bot_ratio, echo_density, depth]`
- Loads threshold from model metadata (`metrics.json`) when available
- Returns prediction, confidence, and threshold used

### Explainability
- Computes SHAP attributions over runtime feature vector
- Produces ranked feature importance for `/explain`
- Generates structured explanation for `/analyze` and `/analyze/news`:
  - summary
  - reasoning lines
  - risk level
  - top drivers
  - recommendation

### Visualization
- Next.js UI pages for generation, simulation, analysis, and real-time news triage
- Animated graph visualization (`three.js`) driven by graph statistics and propagation metrics
- Live simulation control panel with debounced updates

## End-to-End Workflow

1. **Input Acquisition**
   - User enters topic (`/generate`, `/analyze`) or selects a live headline (`/news-analyst`)
2. **Narrative Construction**
   - Backend generates text via Ollama (`llm_service`) or local fallback text policy
3. **Graph Simulation**
   - `simulation_service` builds a graph from text length and seeded randomness
   - Assigns agent roles (`normal`, `influencer`, `bot`, `skeptic`)
   - Simulates multi-step exposure spread
4. **Metric Extraction**
   - `propagation_service` computes:
     - `reach`
     - `depth`
     - `velocity`
     - `velocity_by_step`
     - `echo_chamber_density`
   - Graph stats include node/edge counts and agent ratios
5. **Feature Transformation**
   - `feature_utils.metrics_to_feature_vector` maps to `[velocity, bot_ratio, echo_density, depth]`
6. **Inference**
   - `model_service` predicts class probability and applies threshold
7. **Attribution + Narrative Explanation**
   - `xai_service` computes SHAP values
   - `explanation_engine` produces human-readable rationale and risk
8. **Frontend Reporting**
   - UI displays label, confidence, metrics, XAI drivers, and recommendation

## Architecture

### Frontend (Next.js 14 + TypeScript)
- Responsibilities:
  - user input capture
  - API orchestration via `frontend/lib/api.ts`
  - visualization and analyst-focused reporting
- Primary pages:
  - `/` home overview
  - `/generate`
  - `/simulate`
  - `/analyze`
  - `/news-analyst`

### Backend (FastAPI)
- Responsibilities:
  - exposes REST endpoints
  - orchestrates generation/simulation/prediction/explanation
  - loads inference artifacts at startup
  - handles news ingestion normalization
- Startup:
  - `backend/run.py` launches Uvicorn
  - `backend/app/main.py` wires routers and model service preload

### AI/ML Pipeline (Offline Training + Online Inference)
- Offline (`AI/main.py`):
  - dataset discovery
  - loading FakeNewsNet/LIAR/PHEME data
  - preprocessing
  - graph construction
  - feature construction
  - GAT training/evaluation/export
- Online (`backend/app/services/model_service.py`):
  - consumes exported artifacts (`AI/outputs/*`)
  - predicts from compact runtime features

## Evaluation status and reproducibility

Numeric evaluation claims are intentionally absent from this README. The currently
tracked generated artifacts cannot be reproduced from the repository: their dataset
registry is empty and their artifact paths refer to a different machine. Treat them
as stale local files, not as valid evidence; remove them from version control before
the next commit (they are now ignored for future runs).

To produce results, place supported data under `AI/dataset/`, confirm that
`discover_datasets()` reports the files, then run `AI/main.py`. A successful run
writes the dataset registry, fixed train/validation/test split indices, metrics, and
an `artifact_manifest.json` containing the registry fingerprint and feature schema.
Do not compare, publish, or serve artifacts whose manifest does not match the data
used for the run.

The runtime features (`velocity`, `bot_ratio`, `echo_density`, `depth`) are derived
from the simulator, while offline tabular baseline features are dataset-derived.
Those are different data-generating processes and must not be presented as a
like-for-like model comparison.

## Core Concepts

### Graph Neural Networks (GNNs)
A GNN learns from both node content and connectivity. In ACWS, graph signals encode who interacts with whom and how information cascades through structure. The model architecture is a residual multi-block Graph Attention Network (`GraphGATClassifier`).

### Propagation Modeling
Propagation is simulated as iterative exposure spread over a synthetic social graph. Each step activates neighbors based on susceptibility and credibility-weighted sharing probability.

### Feature Engineering
Runtime detection uses four features:
- `velocity`: average new exposures per step
- `bot_ratio`: share of bot-type agents in simulated graph
- `echo_density`: fraction of intra-community edges
- `depth`: cascade depth (max propagation step)

These features represent spread dynamics rather than text semantics alone.

### SHAP / Explainable AI
SHAP estimates per-feature contribution to model output. ACWS uses these values to:
- rank influential features
- infer contribution direction (toward misinformation vs organic spread)
- generate structured analyst-facing explanations

## Backend Design

### Service Layer
- `llm_service.py`
  - Sends prompts to Ollama (`OLLAMA_URL`)
  - Provides typed failure handling and local fallback text mode
- `simulation_service.py`
  - Builds graph
  - assigns agent properties
  - runs spread simulation
  - supports optional runtime override knobs
- `model_service.py`
  - Loads model artifacts
  - performs inference
  - resolves threshold and confidence
- `xai_service.py`
  - Builds SHAP explainer with background matrix
  - returns feature ranking and raw attribution payload
- `explanation_engine.py`
  - Merges metrics + SHAP into summary, reasoning, risk tier, key drivers, recommendation
- `news_service.py`
  - Fetches NewsAPI/GDELT data
  - normalizes articles to `{title, description, source, url}`
  - provides mock fallback when external sources fail

### API Endpoints and Purpose
- `POST /generate/misinformation`: generate misinformation-style text
- `POST /generate/counter`: generate corrective narrative
- `POST /generate/neutral`: generate neutral report
- `POST /simulate`: simulate spread and return graph/propagation metrics
- `POST /predict`: classify feature vector
- `POST /explain`: return ranked SHAP feature importance
- `POST /analyze`: topic -> generated text -> simulation -> detection -> explanation
- `POST /analyze/news`: selected headline context -> full analysis pipeline
- `GET /news/trending`: fetch normalized trending news list
- `GET /news/search`: fetch normalized news by topic

## Frontend Design

### Pages
- `frontend/app/page.tsx`
  - home landing with capability summary
- `frontend/app/generate/page.tsx`
  - topic input + narrative generation actions
  - trending topic click-to-fill support
- `frontend/app/simulate/page.tsx`
  - text-based simulation
  - slider controls for steps, bot ratio, initial seeds, influence strength
  - 300ms debounced live updates after initial run
- `frontend/app/analyze/page.tsx`
  - full end-to-end analysis report with graph, metrics, confidence, explanation
- `frontend/app/news-analyst/page.tsx`
  - title-only trending list
  - selected news analysis
  - XAI reasoning and feature drivers presentation

### Key Components
- `GraphVisualization.tsx`
  - animated 3D network rendering with propagation-driven activation behavior
- `ResultCard.tsx`
  - compact metric and section card abstraction
- `Navbar.tsx`
  - primary navigation across analysis flows

## Data and Features

### Training Data Sources
- FakeNewsNet-style files
- LIAR benchmark TSV splits
- PHEME-style rumor CSV candidates

### Preprocessing and Graph Construction
- Required canonical columns: `id`, `parent_id`, `text`, `label`, `timestamp`
- Text normalization removes URLs/punctuation and standardizes casing
- Graph contains:
  - news nodes
  - user nodes
  - user-news edges
  - user-user edges
  - parent-child conversational edges

### Feature Sets
- **Node features (training):**
  - text embeddings (hashing/transformer/sentence-transformer)
  - structural features (degree, centrality, clustering, propagation depth)
  - temporal features
  - bot probability
  - optional compressed user feature vectors (FakeNewsNet user `.mat` features;
    LIAR speaker truthfulness-history counts, party, and hashed speaker/subject —
    28 dims, see `AI/src/gnn_pipeline/loaders.py::_build_liar_speaker_history_features`)
- **Backend runtime features (inference):**
  - velocity
  - bot ratio
  - echo density
  - depth

## Model and Inference

### Model
- Architecture: residual Graph Attention classifier (`GraphGATClassifier`),
  defined once in `AI/src/gnn_pipeline/model.py` and imported by both the
  offline training pipeline and `backend/app/services/model_service.py` (the
  backend adds `AI/src` to `sys.path` in `backend/app/core/config.py`) — there
  is no longer a second hand-synced copy of the architecture to drift out of
  sync
- Trained offline via `AI/src/gnn_pipeline/train.py`
- Artifacts exported to `AI/outputs`:
  - `model_state_dict.pt`
  - `model_torchscript.pt`
  - `processed_graph.pt`
  - `metrics.json`

### Inference Path
- Runtime vector is built from simulation metrics
- Probability is produced by loaded model path
- Binary label uses `probability > threshold_used`

### Threshold Logic
- Threshold is read from `metrics.json` if available
- Fallback defaults are applied when metadata is unavailable

### Output Interpretation
- `prediction=1`: misinformation-like propagation behavior
- `prediction=0`: comparatively organic spread behavior
- `confidence`: max(probability, 1-probability)

## Results

Results come from isolated per-dataset graphs using real sentence-transformer
text embeddings (`sentence-transformers/all-MiniLM-L6-v2`), seeded 70/15/15
splits persisted to disk, and the same residual GAT profile trained for up to
60 epochs with early stopping (patience 10) on the validation F1. The live
backend uses the FakeNewsNet model by default because it has the richest graph
structure; LIAR and PHEME remain available for comparison.

| Dataset | Accuracy | F1 | ROC-AUC |
| --- | ---: | ---: | ---: |
| FakeNewsNet | 0.7174 | 0.8267 | 0.4040 |
| LIAR | 0.7176 | 0.8155 | 0.7556 |
| PHEME | 0.9348 | 0.8541 | 0.9759 |
| **Macro-average** | **0.7899** | **0.8321** | **0.7118** |

LIAR's graph is intentionally edge-free because its source statements do not
provide user/news relationships; that structural difference is unchanged and
still a controlled comparison, not an error hidden inside a merged graph.
LIAR's node features now include the speaker truthfulness-history counts
(`barely_true`, `false_count`, `half_true_count`, `mostly_true_count`,
`pants_fire_count`), party, and a hashed speaker/subject signal, which the
loader previously discarded even though the LIAR literature shows this
metadata carries more signal than the statement text alone (see
`CHANGELOG.md`). Adding them moved LIAR's accuracy from 0.6566 to 0.7176 and
ROC-AUC from 0.6193 to 0.7556 — all of that improvement comes from richer node
features, not connectivity, since the graph itself is still edge-free by
dataset construction.

FakeNewsNet's ROC-AUC (0.40) looks worse than random despite a healthy
accuracy/F1, which is a small-sample artifact, not a broken model: its test
split is only 46 examples (14 negative), so a couple of probability-ranking
flips swing the AUC estimate sharply. Treat FakeNewsNet's ranking metrics as
high-variance and the thresholded metrics (accuracy/F1, which is what the
backend actually uses to decide) as the more meaningful ones for this dataset.
PHEME, with a 9,367-example test split, gives the most statistically stable
estimate of the three, and it's also the strongest result (ROC-AUC 0.976).

Per-dataset metrics and plots are stored under each dataset output folder;
`AI/outputs/macro_eval_metrics.json` contains the unweighted macro-average.

### GCN Baseline

The same isolated graphs and persisted splits were used for a four-layer
residual GCN baseline:

| Model | FakeNewsNet F1 | LIAR F1 | PHEME F1 | Macro-average F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression surrogate | 0.8205 | 0.7862 | 0.3626 | 0.6564 |
| GCN baseline | 0.8205 | 0.8052 | 0.8149 | 0.8135 |
| GraphGATClassifier (ours) | 0.8267 | 0.8155 | 0.8541 | **0.8321** |

The GAT is the strongest model overall and clearly ahead of both baselines on
PHEME (the dataset with enough data — 9,367 test examples — for the comparison
to be statistically meaningful), while GAT/GCN/LR are close on FakeNewsNet,
where sample size dominates any architectural difference. The LR surrogate
lagging badly on PHEME (0.36 vs. GAT 0.85) shows the 4 hand-engineered runtime
features alone are not sufficient there — the graph structure and richer node
features are doing real work. The LR surrogate's LIAR gap (0.7862 vs. GAT's
0.8155) is the same story on a smaller scale: the LR surrogate only ever sees
the 4 runtime features, so it does not benefit from the speaker
truthfulness-history node features described above, while the GAT and GCN do.
These comparisons are limited by the small FakeNewsNet sample and the
different graph structures across datasets. Detailed baseline files are stored
in `AI/outputs/gcn_macro_metrics.json` and `AI/outputs/lr_macro_metrics.json`.

### Serving-Model Ablation

The leave-one-out ablation targets the default FakeNewsNet serving model and
uses its persisted test features. All four single-feature ablations still
produced the same F1 (`0.8205`, delta `0.0000`) as before. This is a structural
property of the serving path, not a training artifact: `TrainedGNNDetectionModel`
folds the 4 runtime features into a handful of dimensions of a 384+ dimensional
graph-feature template (the rest of the template is the dataset's frozen mean
vector), and at a 46-example test size that's not enough signal to move any
single feature's ablation past the decision boundary. SHAP mean absolute
importance ranked the features as depth (`0.0083`), velocity (`0.0058`), bot
ratio (`0.0049`), and echo density (`0.0044`). Full output is in
`AI/outputs/fakenewsnet/eval/ablation.json`.

## Explainability

### SHAP Integration
- `xai_service` creates a SHAP explainer around `predict_positive_probabilities`
- Returns both:
  - ranked absolute feature importance (`/explain`)
  - raw per-feature SHAP values for structured reports

### Explanation Synthesis
- `explanation_engine` combines:
  - SHAP ranking
  - propagation thresholds
  - model confidence
- Produces:
  - summary text
  - reasoning bullets
  - risk level (`low|medium|high`)
  - top feature drivers with directional impact
  - recommendation

`risk_level` is model-derived, not a hand-written combination of raw feature
thresholds: `train.py` computes tertile cut points (`risk_thresholds.low_medium`,
`risk_thresholds.medium_high`) over the validation-set probability distribution
at the best epoch and persists them in `metrics.json`; `explanation_engine`
buckets the live prediction probability against those thresholds. Tertiles were
chosen over probability calibration (e.g. Platt/isotonic scaling) because
FakeNewsNet's test split is only 46 examples — not enough to fit a calibration
curve reliably. A model trained before this was added falls back to plain
thirds (`0.33`/`0.67`).

### Frontend Presentation
- Analyze pages display:
  - confidence and risk band
  - reasoning list
  - top feature drivers and importance
  - recommendation text

## Folder Structure

```text
AI-driver-cognitive-warfare-simulator/
├── AI/
│   ├── main.py
│   ├── requirements.txt
│   ├── dataset/
│   ├── outputs/
│   └── src/gnn_pipeline/
├── backend/
│   ├── run.py
│   ├── requirements.txt
│   ├── .env
│   └── app/
│       ├── main.py
│       ├── core/
│       ├── routes/
│       ├── schemas/
│       └── services/
├── frontend/
│   ├── package.json
│   ├── app/
│   ├── components/
│   ├── lib/
│   └── styles/
└── README.md
```

## Setup Instructions

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- Ollama (for LLM-backed generation)

### 1) Backend Setup
```bash
cd backend
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python run.py
```

Backend runs at `http://localhost:8000`.

### 2) Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:3000`.

### 3) AI Pipeline Setup (Training/Artifact Regeneration)
```bash
cd AI
pip install -r requirements.txt
python main.py --dataset-root ./dataset --output-dir ./outputs
```

### Environment Variables
- Backend (`backend/.env`)
  - `OLLAMA_MODEL` (optional)
  - `ALLOW_LLM_FALLBACK` (optional)
  - `NEWS_API_KEY` (required for NewsAPI usage)
- Frontend (`frontend/.env.local`, optional)
  - `NEXT_PUBLIC_API_URL=http://localhost:8000`

## Usage Guide

### Run the System
1. Start backend
2. Start frontend
3. Open frontend in browser

### Primary Workflows
- **Generate**
  - Provide topic
  - choose narrative mode
  - inspect generated text
- **Simulate**
  - Provide narrative text
  - tune controls (`steps`, `bot ratio`, `initial seeds`, `influence strength`)
  - inspect graph and propagation metrics
- **Analyze**
  - Provide topic
  - run full pipeline report
  - review classification + XAI
- **News Analyst**
  - load trending headlines
  - select title
  - run contextual analysis using title + description payload

### Example API Calls
```bash
curl http://127.0.0.1:8000/
```

```bash
curl -X POST http://127.0.0.1:8000/simulate \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"sample narrative\",\"steps\":10,\"bot_ratio_override\":0.25,\"seed_nodes\":5,\"influencer_strength\":0.8}"
```

```bash
curl -X GET "http://127.0.0.1:8000/news/trending"
```

## Limitations

- Propagation is simulated, not learned from live platform event streams
- Runtime detector uses a compact engineered feature vector (4 features), which limits representational richness at inference time
- SHAP explanations depend on background data assumptions and are approximate
- News ingestion reliability depends on external API availability and key configuration
- LLM generation quality and style vary with local model and runtime conditions

## Future Improvements

- Add stream-based ingestion connectors for real social/news event timelines
- Introduce temporal/heterogeneous GNN variants for richer interaction modeling
- Add persistent experiment tracking and model version registry
- Implement calibration and uncertainty reporting for risk probabilities
- Expand multilingual pipeline support across preprocessing, generation, and evaluation

## Conclusion

ACWS implements a complete, modular pipeline for misinformation scenario analysis: narrative generation or ingestion, propagation simulation, model-based risk detection, and explainable reporting. The system is structured for experimentation and analyst workflows, with clear separation between offline model development and online inference orchestration.
