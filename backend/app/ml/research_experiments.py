"""Controlled, non-destructive temporal-feature and robustness experiments.

This runner writes only into a new timestamped outputs/research_experiments folder.
It does not save models or change the application's current prediction artifacts.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from app.config import OUTPUT_DIR
from app.services import data_service
from app.ml.temporal import FEATURES, training_samples
from app.ml.train import (
    STATIC_FEATURES,
    _binary_calibration_metrics,
    _multiclass_calibration_metrics,
    _xgb_params,
)

WINDOW_FEATURES = {
    window: [f"{prefix}_{window}m" for prefix in ("R", "F", "M", "Basket")]
    for window in (3, 6, 12)
}
TEMPORAL_RFM = [f"{prefix}_{window}m" for window in (3, 6, 12) for prefix in ("R", "F", "M")]
TEMPORAL_PLUS_BASKET = TEMPORAL_RFM + [f"Basket_{window}m" for window in (3, 6, 12)]
ABLATIONS = {
    "static_rfm_xgboost": STATIC_FEATURES,
    "temporal_rfm_xgboost": TEMPORAL_RFM,
    "temporal_rfm_plus_basket_xgboost": TEMPORAL_PLUS_BASKET,
    "full_current_features_xgboost": FEATURES,
}


def _scores(y_true, y_pred, labels, names):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": [
            {"class": names[i], "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support))
        ],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_order": list(names),
    }


def _train_rows(frame: pd.DataFrame, target: str, target_start: pd.Period) -> pd.DataFrame:
    """Keep only labels observable before the first test snapshot."""
    eligible = frame.dropna(subset=[target]).copy()
    eligible = eligible[eligible["Month"] < target_start]
    if target == "NextSegment":
        return eligible[eligible["NextMonth"] < target_start].copy()
    test_cutoff = target_start.start_time
    ends = eligible["Month"].map(lambda month: (month + 1).start_time + timedelta(days=90))
    return eligible[ends <= test_cutoff].copy()


def _test_rows(frame: pd.DataFrame, target: str, start: pd.Period, end: pd.Period) -> pd.DataFrame:
    return frame[
        frame["Month"].between(start, end)
        & frame[target].notna()
    ].copy()


def _segments(train, test, features, include_legacy_logistic=False):
    original_test_rows = len(test)
    encoder = LabelEncoder().fit(train["NextSegment"].astype(str))
    train = train[train["NextSegment"].astype(str).isin(encoder.classes_)].copy()
    test = test[test["NextSegment"].astype(str).isin(encoder.classes_)].copy()
    if len(encoder.classes_) < 2 or test.empty:
        raise ValueError("Next-segment split has fewer than two learned classes or no comparable test rows.")
    y_train = encoder.transform(train["NextSegment"].astype(str))
    y_test = encoder.transform(test["NextSegment"].astype(str))
    x_train, x_test = train[features].fillna(0), test[features].fillna(0)
    model = _xgb_params(len(encoder.classes_))
    model.fit(x_train, y_train)
    pred = model.predict(x_test).astype(int)
    probabilities = model.predict_proba(x_test)
    labels = np.arange(len(encoder.classes_))
    result = _scores(y_test, pred, labels, encoder.classes_.tolist())
    result.update(_multiclass_calibration_metrics(y_test, probabilities))
    result["train_rows"] = int(len(train))
    result["test_rows"] = int(len(test))
    result["test_class_counts"] = test["NextSegment"].astype(str).value_counts().reindex(encoder.classes_, fill_value=0).astype(int).to_dict()
    result["train_class_counts"] = train["NextSegment"].astype(str).value_counts().reindex(encoder.classes_, fill_value=0).astype(int).to_dict()
    result["classes_unseen_in_train_excluded_from_test"] = int(original_test_rows - len(test))
    if include_legacy_logistic:
        baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
        baseline.fit(train[STATIC_FEATURES].fillna(0), y_train)
        legacy_pred = baseline.predict(test[STATIC_FEATURES].fillna(0))
        legacy_prob = baseline.predict_proba(test[STATIC_FEATURES].fillna(0))
        legacy = _scores(y_test, legacy_pred, labels, encoder.classes_.tolist())
        legacy.update(_multiclass_calibration_metrics(y_test, legacy_prob))
        result["legacy_static_rfm_logistic"] = legacy
    return result


def _persistence(test, labels):
    names = list(labels)
    known = test["NextSegment"].astype(str).isin(names)
    eligible = test.loc[known]
    y_true = eligible["NextSegment"].astype(str).tolist()
    y_pred = eligible["Segment"].astype(str).where(eligible["Segment"].astype(str).isin(names), "__unknown__").tolist()
    return {
        "test_rows": int(len(eligible)),
        "classes_unseen_in_train_excluded_from_test": int(len(test) - len(eligible)),
        "test_class_counts": eligible["NextSegment"].astype(str).value_counts().reindex(names, fill_value=0).astype(int).to_dict(),
        **_scores(y_true, y_pred, names, names),
        "definition": "Predict the current snapshot segment again as the next snapshot segment.",
    }


def _churn(train, test, features, include_legacy_logistic=False):
    train = train.dropna(subset=["Churn90d"]).copy()
    test = test.dropna(subset=["Churn90d"]).copy()
    if train["Churn90d"].nunique() < 2 or test["Churn90d"].nunique() < 2:
        raise ValueError("Inactivity split requires both classes in train and test after the 90-day eligibility filter.")
    y_train, y_test = train["Churn90d"].astype(int), test["Churn90d"].astype(int)
    x_train, x_test = train[features].fillna(0), test[features].fillna(0)
    model = _xgb_params(2, binary=True)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    pred = (probabilities >= 0.5).astype(int)
    result = _scores(y_test, pred, [0, 1], ["No 90-day inactivity", "90-day inactivity"])
    result.update(_binary_calibration_metrics(y_test, probabilities))
    result["roc_auc"] = float(roc_auc_score(y_test, probabilities))
    result["pr_auc"] = float(average_precision_score(y_test, probabilities))
    result["decision_threshold"] = 0.5
    result["train_rows"] = int(len(train))
    result["test_rows"] = int(len(test))
    result["positive_rate_train"] = float(y_train.mean())
    result["positive_rate_test"] = float(y_test.mean())
    result["train_class_counts"] = y_train.value_counts().reindex([0, 1], fill_value=0).astype(int).to_dict()
    result["test_class_counts"] = y_test.value_counts().reindex([0, 1], fill_value=0).astype(int).to_dict()
    if include_legacy_logistic:
        baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
        baseline.fit(train[STATIC_FEATURES].fillna(0), y_train)
        legacy_prob = baseline.predict_proba(test[STATIC_FEATURES].fillna(0))[:, 1]
        legacy_pred = (legacy_prob >= 0.5).astype(int)
        legacy = _scores(y_test, legacy_pred, [0, 1], ["No 90-day inactivity", "90-day inactivity"])
        legacy.update(_binary_calibration_metrics(y_test, legacy_prob))
        legacy["roc_auc"] = float(roc_auc_score(y_test, legacy_prob))
        legacy["pr_auc"] = float(average_precision_score(y_test, legacy_prob))
        result["legacy_static_rfm_logistic"] = legacy
    return result


def _test_windows(months, target):
    months = sorted(months)
    if target == "NextSegment":
        latest = months[max(0, int(len(months) * 0.8)):]
        candidates = [months[max(6, int(len(months) * 0.50)):][:3], months[max(6, int(len(months) * 0.68)):][:3], latest]
    else:
        candidates = [months[max(6, int(len(months) * 0.50)):][:2], months[max(6, int(len(months) * 0.70)):][:2], months[-2:]]
    unique = []
    seen = set()
    for window in candidates:
        window = [month for month in window if month in months]
        if not window:
            continue
        key = (window[0], window[-1])
        if key not in seen:
            seen.add(key)
            unique.append(window)
    return unique


def _run_target(frame, target, feature_groups, test_windows, *, robustness=False):
    runs = []
    for window in test_windows:
        start, end = window[0], window[-1]
        before_boundary = frame[frame["Month"] < start]
        observable = before_boundary.dropna(subset=[target])
        train = _train_rows(frame, target, start)
        test = _test_rows(frame, target, start, end)
        if target == "Churn90d":
            train = train.dropna(subset=[target])
            test = test.dropna(subset=[target])
        if train.empty or test.empty:
            continue
        run = {"test_period": [str(start), str(end)], "target": target,
               "raw_train_candidates": int(len(before_boundary)),
               "incomplete_or_unlabeled_train_rows": int(len(before_boundary) - len(observable)),
               "purged_for_temporal_overlap": int(len(observable) - len(train)),
               "train_rows": int(len(train)), "test_rows": int(len(test)), "models": {}}
        if target == "NextSegment":
            classes = sorted(train["NextSegment"].astype(str).unique().tolist())
            run["test_class_counts_before_model_filter"] = test["NextSegment"].astype(str).value_counts().to_dict()
            run["models"]["persistence"] = _persistence(test, classes)
            for name, features in feature_groups.items():
                result = _segments(train, test, features, include_legacy_logistic=not robustness)
                result["features"] = features
                run["models"][name] = result
        else:
            run["test_class_counts"] = test[target].astype(int).value_counts().reindex([0, 1], fill_value=0).to_dict()
            for name, features in feature_groups.items():
                result = _churn(train, test, features, include_legacy_logistic=not robustness)
                result["features"] = features
                run["models"][name] = result
        runs.append(run)
    return runs


def main():
    samples = training_samples()
    if samples.empty:
        raise SystemExit("No temporal samples could be constructed from the configured dataset.")
    keys = ["Customer ID", "Month"]
    duplicate_count = int(samples.duplicated(keys, keep=False).sum())
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} rows in duplicated customer-month states; refusing to evaluate ambiguous samples.")
    required = set(STATIC_FEATURES + FEATURES + ["NextSegment", "Churn90d", "Month", "NextMonth", "Segment"])
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"Experiment samples lack required columns: {sorted(missing)}")

    next_months = sorted(samples.dropna(subset=["NextSegment"])["Month"].unique())
    churn_months = sorted(samples.dropna(subset=["Churn90d"])["Month"].unique())
    next_windows = _test_windows(next_months, "NextSegment")
    churn_windows = _test_windows(churn_months, "Churn90d")
    next_runs = _run_target(samples.dropna(subset=["NextSegment"]).copy(), "NextSegment", ABLATIONS, next_windows)
    churn_runs = _run_target(samples, "Churn90d", ABLATIONS, churn_windows)

    window_groups = {f"lookback_{window}m": WINDOW_FEATURES[window] for window in (3, 6, 12)}
    window_groups["all_lookbacks_3_6_12m"] = TEMPORAL_PLUS_BASKET
    window_next = _run_target(samples.dropna(subset=["NextSegment"]).copy(), "NextSegment", window_groups, next_windows[-1:], robustness=True)
    window_churn = _run_target(samples, "Churn90d", window_groups, churn_windows[-1:], robustness=True)

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = OUTPUT_DIR / "research_experiments" / f"run_{stamp}"
    destination.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    experiment_data_path = data_service.DATA_PATH
    with experiment_data_path.open("rb") as workbook:
        for chunk in iter(lambda: workbook.read(1024 * 1024), b""):
            digest.update(chunk)
    from app.services.data_service import purchase_rows
    last_purchase = purchase_rows()["InvoiceDate"].max().date().isoformat()
    llm_configured = bool(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))
    result = {
        "experiment_timestamp_utc": timestamp,
        "dataset": str(experiment_data_path),
        "dataset_sha256": digest.hexdigest(),
        "dataset_last_purchase": last_purchase,
        "sample_rows": int(len(samples)),
        "unique_customers": int(samples["Customer ID"].nunique()),
        "unique_customer_months": int(samples[keys].drop_duplicates().shape[0]),
        "duplicate_customer_month_rows": duplicate_count,
        "feature_groups": ABLATIONS,
        "protocol": {
            "estimators": "Same XGBoost configuration (250 trees, depth 4, learning rate 0.05, seed 42) across feature ablations; legacy balanced logistic RFM retained as a separate comparison on canonical splits.",
            "next_segment_target": "Same customer's segment in the immediately following observed monthly snapshot.",
            "inactivity_target": "No positive purchase in [snapshot cutoff, cutoff + 90 days); only fully observed horizons are scored.",
            "purge": "For next-segment, training targets must precede the first test source month. For inactivity, training's 90-day outcome horizon must end on or before the first test snapshot cutoff.",
            "test_populations": "All feature ablations within a given target and split use the same row population; rows with next-segment classes absent from training are excluded for every model and reported.",
            "probabilities": "Raw probabilities; calibration metrics are diagnostics and no post-hoc calibrator is fitted.",
            "alternative_lookbacks": "Evaluate 3-month, 6-month, and 12-month RFM plus basket feature groups, and a combined 3/6/12-month feature set on the latest eligible windows.",
        },
        "temporal_audit": {
            "unique_next_segment_months": [str(value) for value in next_months],
            "unique_fully_observed_inactivity_months": [str(value) for value in churn_months],
            "next_segment_runs": next_runs,
            "inactivity_runs": churn_runs,
            "lookback_robustness_next_segment": window_next,
            "lookback_robustness_inactivity": window_churn,
        },
        "recommendation_evaluation": {
            "status": "not run; LLM is configured but no paired evaluation has been run" if llm_configured else "not run; no LLM API key configured",
            "proposed_protocol": "Freeze customer evidence and prompt version; generate rule-based and LLM recommendations from identical evidence; blind raters to provider; rate grounding, relevance, specificity, and actionability on a predefined 1–5 anchored rubric; report per-dimension scores, agreement, and failures. No business impact is inferred without a randomized intervention study.",
        },
        "limitations": [
            "Single historical retailer dataset ending in 2011; no cross-retailer or current-market validation.",
            "Relative monthly RFM quintiles are heuristic targets whose boundaries and meaning can vary over time.",
            "Repeated customers occur in train and later test periods; estimates describe temporal continuation for a partially known customer population, not cold-start customers.",
            "The inactivity label is an observed-data proxy and not contractual churn.",
            "A single dataset cannot establish business impact or causal effectiveness of retention actions.",
        ],
    }
    output_path = destination / "ablation_results.json"
    output_path.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({"result_path": str(output_path), "timestamp": timestamp,
                      "sample_rows": len(samples), "duplicate_customer_month_rows": duplicate_count,
                      "next_segment_splits": len(next_runs), "inactivity_splits": len(churn_runs),
                      "latest_next_segment_period": next_runs[-1]["test_period"] if next_runs else None,
                      "latest_inactivity_period": churn_runs[-1]["test_period"] if churn_runs else None}, indent=2))


if __name__ == "__main__":
    main()
