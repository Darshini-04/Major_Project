# PulseIQ

PulseIQ is a local research prototype for temporal e-commerce customer analytics. It reads the two-sheet Online Retail II workbook, builds relative RFM and monthly customer profiles, and provides a Next.js dashboard with a FastAPI data API. Chronological next-segment and 90-day inactivity models, holdout metrics, customer predictions, and SHAP outputs are available in the current workspace.

## Data setup

The Online Retail II workbook is included at `data/raw/online_retail_II.xlsx`. The backend uses this path by default. Copy `.env.example` to `.env` only if you want to override `PULSEIQ_DATA_PATH` with a different workbook location.

The source columns required are `Invoice`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, and `Country`. Positive purchase features exclude cancelled invoices (invoice number begins with `C`), non-positive quantities or prices, and rows missing required customer/date fields. Raw rows remain unmodified in memory for transparency.

## Run locally

Python 3.12 was used for the initial prototype. Create a virtual environment and install backend dependencies:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

In another terminal:

```bash
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

Open `http://localhost:3000`. The API health endpoint is `http://localhost:8000/api/health`.

Run the real data pipeline and its focused tests from the repository root:

```bash
backend/.venv/bin/python backend/scripts/run_pipeline.py
PYTHONPATH=backend backend/.venv/bin/python -m unittest discover -s backend/tests -v
```

After the pipeline dependencies are installed, build the temporal labels, train chronological baselines and XGBoost models, save evaluation metrics and current predictions, and create SHAP summaries with:

```bash
PYTHONPATH=backend backend/.venv/bin/python -m app.ml.train
```

The model-training step can take several minutes. It writes model artifacts under `models/` and measured outputs under `outputs/`.

## Current methodology and limits

The initial RFM profile uses the last observed date plus one day as its reference point. Frequency counts distinct invoices; monetary value is quantity multiplied by unit price. Segment scores are relative quintiles in this dataset, and the names are heuristic. They are not calibrated business rules. Refunds and cancellations are excluded from positive-purchase RFM features. Currency values are displayed in GBP based on the dataset convention; currency conversion is not performed.

The data pipeline writes a static dashboard snapshot to `frontend/public/analytics_snapshot.json`. It includes monthly sales, country and product rankings, returns and cancellations context, and customer profiles. The snapshot is data-only; connect FastAPI for current model outputs, SHAP explanations, evaluation results, prediction filtering, and downloads. Retention Advisor uses a deterministic evidence-based fallback unless a configured LLM provider responds successfully.

The evaluation report records the chronological split, labels, feature list, model settings, per-class scores, confusion matrices, Brier scores, and calibration errors. These calibration values are diagnostics only: the current customer-level probabilities are raw model scores and have not been transformed by a post-hoc calibrator. Download the JSON report from the Model Evaluation view or `GET /api/model-evaluation/download`. Download customer-level scores from Prediction & churn or `GET /api/predictions/export.csv`.

When a model has been trained, retention suggestions can also use its predicted segment, probabilities, and SHAP evidence. An optional OpenAI-compatible chat completions endpoint can be configured with `LLM_API_KEY`, `LLM_API_BASE`, and `LLM_MODEL`; only a minimal structured profile and evidence are sent. Without a key or when the provider request fails, the deterministic fallback is used. The LLM drafts actions and does not predict churn.

## Project structure

- `backend/app/`: API, data preparation, temporal profiling, segmentation, and recommendation services.
- `frontend/`: Next.js dashboard.
- `data/`: local input and processed data locations.
- `models/`: saved model artifacts after validated training.
- `outputs/`: experimental results after reproducible evaluation.
- `docs/`: architecture, research outline, and experimental report.

See `IMPLEMENTATION_STATUS.md` for completed and pending work.
