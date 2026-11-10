# PulseIQ implementation status

## Completed

- Existing Next.js 15.5.28 frontend and FastAPI backend remain in place, with all eight navigation areas preserved.
- Online Retail II ingestion, positive-purchase filtering, RFM profiles, monthly journeys, observed transitions, and rule-based retention recommendations remain available.
- Chronological static RFM and temporal XGBoost models are trained. Current model artifacts and customer-level predictions are under `models/` and `outputs/`.
- `outputs/model_metrics.json` includes comparable holdout results, confusion matrices, per-class scores, calibration diagnostics, split dates and sizes, label definitions, model configuration, and experiment timestamp. The prior report is preserved as `outputs/model_metrics_pre_calibration.json`.
- Global next-segment/churn SHAP summaries and local customer-level SHAP contributions are available.
- Dashboard status reflects live model artifacts. Customer Segments includes an integrated intelligence view with RFM, monthly history, prediction, SHAP, and retention advice; it can filter customers with prediction rows.
- Model evaluation JSON and customer prediction CSV exports are available through the UI and API.
- Automated tests cover calibration metrics, next-month transition boundaries, the complete 90-day label horizon, prediction filtering, artifact downloads, segmentation, and recommendation evidence.

## Current measured holdout

- Next segment: train 2009-12–2011-06 (61,731 rows); test 2011-07–2011-11 (20,646 rows). Temporal XGBoost accuracy 0.7056 and macro-F1 0.6865; static RFM accuracy 0.6526 and macro-F1 0.6366.
- 90-day inactivity proxy: train 61,731 rows; test 2011-07–2011-08 (8,368 rows). Temporal XGBoost accuracy 0.7079, ROC-AUC 0.7824, and PR-AUC 0.7699.
- The dataset ends on 2011-12-09. Inactivity is a proxy label, segment names are heuristic, and current probabilities are not post-hoc calibrated. See `docs/experimental_report.md` and the metrics JSON for full results.

## Remaining work

- Evaluate a proper chronological probability-calibration procedure using a separate calibration period before applying any transform to customer predictions.
- Configure and verify an LLM provider if LLM-generated retention advice is desired; the deterministic fallback remains available.
- Review segment definitions and the inactivity proxy with the project supervisor. Results describe this historical dataset and do not establish present-day performance or causal retention impact.
