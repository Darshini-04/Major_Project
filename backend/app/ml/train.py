"""Chronological model evaluation and final fit for next-segment/churn tasks."""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from app.config import OUTPUT_DIR, ROOT
from app.ml.temporal import FEATURES, training_samples
from app.ml.temporal import temporal_snapshots

MODEL_DIR = ROOT / "models"
STATIC_FEATURES = ["R_12m", "F_12m", "M_12m"]


def _split_by_time(frame: pd.DataFrame):
    months = sorted(frame["Month"].dropna().unique())
    if len(months) < 5: raise ValueError("At least five monthly snapshots with labels are required for chronological evaluation.")
    cut = max(1, int(len(months) * 0.8))
    train_months, test_months = months[:cut], months[cut:]
    return frame[frame.Month.isin(train_months)].copy(), frame[frame.Month.isin(test_months)].copy()


def _xgb_params(classes: int, *, binary: bool = False):
    params = dict(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
                  reg_lambda=1.0, random_state=42, n_jobs=4, tree_method="hist")
    if binary:
        return XGBClassifier(**params, objective="binary:logistic", eval_metric="logloss")
    return XGBClassifier(**params, objective="multi:softprob", num_class=classes, eval_metric="mlogloss")


def _class_metrics(y_true, y_pred, labels):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "macro_precision": float(np.mean(precision)),
            "macro_recall": float(np.mean(recall)), "macro_f1": float(np.mean(f1)),
            "per_class": [{"class": str(c), "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)} for c,p,r,f,s in zip(labels,precision,recall,f1,support)],
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(), "class_order": [str(x) for x in labels]}


def _calibration_bins(confidence, correct, bins: int = 10):
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = []
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & ((confidence <= upper) if index == bins - 1 else (confidence < upper))
        if not mask.any():
            continue
        result.append({"lower": float(lower), "upper": float(upper), "count": int(mask.sum()),
                       "mean_confidence": float(confidence[mask].mean()), "observed_rate": float(correct[mask].mean())})
    return result


def _multiclass_calibration_metrics(y_true, probabilities):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    one_hot = np.eye(probabilities.shape[1])[y_true]
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predicted == y_true).astype(float)
    bins = _calibration_bins(confidence, correct)
    ece = sum(item["count"] * abs(item["mean_confidence"] - item["observed_rate"]) for item in bins) / max(1, len(y_true))
    return {"multiclass_brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
            "top_label_ece": float(ece), "mean_top_class_confidence": float(confidence.mean()), "calibration_bins": bins}


def _binary_calibration_metrics(y_true, probabilities):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    bins = _calibration_bins(probabilities, y_true)
    ece = sum(item["count"] * abs(item["mean_confidence"] - item["observed_rate"]) for item in bins) / max(1, len(y_true))
    return {"brier_score": float(np.mean((probabilities - y_true) ** 2)),
            "expected_calibration_error": float(ece), "mean_predicted_probability": float(probabilities.mean()),
            "calibration_bins": bins}


def _shap_values(model, x: pd.DataFrame, class_index=None):
    raw = shap.TreeExplainer(model).shap_values(x, check_additivity=False)
    if isinstance(raw, list): raw = np.stack(raw, axis=-1)
    raw = np.asarray(raw)
    if raw.ndim == 3:
        if class_index is None:
            selected = raw.mean(axis=2)
        else:
            selected = raw[:, :, int(class_index)]
        return raw, selected
    return raw, raw


def _save_global_shap(model, x: pd.DataFrame, name: str):
    sample = x.sample(min(500, len(x)), random_state=42) if len(x) > 500 else x
    raw, _ = _shap_values(model, sample)
    importance = np.abs(raw).mean(axis=(0, 2)) if raw.ndim == 3 else np.abs(raw).mean(axis=0)
    result = [{"feature": f, "mean_abs_shap": float(v)} for f,v in sorted(zip(x.columns,importance), key=lambda pair:pair[1], reverse=True)]
    (OUTPUT_DIR / f"{name}_shap_importance.json").write_text(json.dumps(result, indent=2))
    return result


def _train_segment(samples: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame):
    ycol = "NextSegment"
    encoder = LabelEncoder().fit(train[ycol].astype(str))
    train = train[train[ycol].astype(str).isin(encoder.classes_)].copy()
    test = test[test[ycol].astype(str).isin(encoder.classes_)].copy()
    if train.empty or test.empty: raise ValueError("Chronological next-segment split has no train or test rows after class validation.")
    y_train = encoder.transform(train[ycol].astype(str)); y_test = encoder.transform(test[ycol].astype(str))
    if len(encoder.classes_) < 2: raise ValueError("Next-segment training data contains fewer than two segment classes.")
    records = {"target":"next_segment", "train_period":[str(train.Month.min()),str(train.Month.max())], "test_period":[str(test.Month.min()),str(test.Month.max())], "train_rows":len(train), "test_rows":len(test), "classes":encoder.classes_.tolist(), "models":{}}
    # Static RFM baseline
    static = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    static.fit(train[STATIC_FEATURES].fillna(0), y_train)
    static_pred = static.predict(test[STATIC_FEATURES].fillna(0))
    static_prob = static.predict_proba(test[STATIC_FEATURES].fillna(0))
    records["models"]["static_rfm_logistic"] = {**_class_metrics(y_test, static_pred, np.arange(len(encoder.classes_))),
                                                    **_multiclass_calibration_metrics(y_test, static_prob)}
    records["models"]["static_rfm_logistic"]["label_names"] = encoder.classes_.tolist()
    joblib.dump(static, MODEL_DIR / "static_rfm_segment.joblib")
    # Temporal feature model
    temporal = _xgb_params(len(encoder.classes_))
    temporal.fit(train[FEATURES].fillna(0), y_train)
    temporal_pred = temporal.predict(test[FEATURES].fillna(0)).astype(int)
    temporal_prob = temporal.predict_proba(test[FEATURES].fillna(0))
    records["models"]["temporal_xgboost"] = {**_class_metrics(y_test, temporal_pred, np.arange(len(encoder.classes_))),
                                                **_multiclass_calibration_metrics(y_test, temporal_prob)}
    records["models"]["temporal_xgboost"]["label_names"] = encoder.classes_.tolist()
    temporal.save_model(str(MODEL_DIR / "next_segment_xgboost.json"))
    joblib.dump(encoder, MODEL_DIR / "next_segment_label_encoder.joblib")
    _save_global_shap(temporal, test[FEATURES].fillna(0), "next_segment")
    return records, encoder


def _train_churn(train: pd.DataFrame, test: pd.DataFrame):
    train = train.dropna(subset=["Churn90d"]).copy(); test = test.dropna(subset=["Churn90d"]).copy()
    if train.Churn90d.nunique() < 2 or test.Churn90d.nunique() < 2:
        return {"status":"unavailable", "reason":"Both classes must be present in chronological train and test periods."}, None
    y_train=train.Churn90d.astype(int); y_test=test.Churn90d.astype(int)
    static=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42))
    static.fit(train[STATIC_FEATURES].fillna(0),y_train)
    xgb=_xgb_params(2,binary=True)
    xgb.fit(train[FEATURES].fillna(0),y_train)
    out={"status":"evaluated", "train_rows":len(train), "test_rows":len(test), "train_period":[str(train.Month.min()),str(train.Month.max())], "test_period":[str(test.Month.min()),str(test.Month.max())], "positive_rate_train":float(y_train.mean()), "positive_rate_test":float(y_test.mean()), "models":{}}
    for name,model,cols in [("static_rfm_logistic",static,STATIC_FEATURES),("temporal_xgboost",xgb,FEATURES)]:
        pred=model.predict(test[cols].fillna(0)).astype(int); prob=model.predict_proba(test[cols].fillna(0))[:,1]
        p,r,f,_=precision_recall_fscore_support(y_test,pred,average="binary",zero_division=0)
        metrics={"accuracy":float(accuracy_score(y_test,pred)),"precision":float(p),"recall":float(r),"f1":float(f),
                 **_class_metrics(y_test,pred,[0,1]), **_binary_calibration_metrics(y_test,prob),
                 "decision_threshold":0.5, "positive_class":"90-day inactivity",
                 "label_names":["No 90-day inactivity", "90-day inactivity"]}
        metrics["roc_auc"] = float(roc_auc_score(y_test,prob)) if y_test.nunique()==2 else None
        metrics["pr_auc"] = float(average_precision_score(y_test,prob)) if y_test.nunique()==2 else None
        out["models"][name]=metrics
    xgb.save_model(str(MODEL_DIR / "churn_xgboost.json")); joblib.dump(static,MODEL_DIR/"static_rfm_churn.joblib")
    _save_global_shap(xgb,test[FEATURES].fillna(0),"churn")
    return out,xgb


def main():
    MODEL_DIR.mkdir(parents=True,exist_ok=True); OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    samples=training_samples()
    if samples.empty: raise SystemExit("No temporal snapshots could be constructed from the configured workbook.")
    next_valid=samples.dropna(subset=["NextSegment"]).copy()
    train,test=_split_by_time(next_valid)
    segment_metrics, encoder=_train_segment(samples,train,test)
    churn_metrics,churn_model=_train_churn(train,test)
    metrics={"dataset":"Online Retail II", "experiment_timestamp_utc":datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z"),
             "evaluation":"Chronological holdout by monthly snapshot (earliest 80% train, latest 20% test); both next-segment baselines share identical periods and rows.",
             "split":{"method":"chronological_month_holdout", "train_fraction":0.8, "next_segment_train_period":segment_metrics["train_period"],
                      "next_segment_test_period":segment_metrics["test_period"], "next_segment_train_rows":segment_metrics["train_rows"],
                      "next_segment_test_rows":segment_metrics["test_rows"], "churn_train_period":churn_metrics.get("train_period"),
                      "churn_test_period":churn_metrics.get("test_period"), "churn_train_rows":churn_metrics.get("train_rows"),
                      "churn_test_rows":churn_metrics.get("test_rows")},
             "labels":{"next_segment":"Segment assigned to the same customer in the immediately following observed monthly snapshot.",
                       "churn_90d":"Proxy label: no positive purchase invoice during the 90 days after the snapshot cutoff; only rows whose complete 90-day horizon ends by the last observed purchase date are labeled.",
                       "segment_names":"Relative per-snapshot RFM quintile labels; heuristic, not business-ground-truth classes."},
             "model_configuration":{"static_rfm_logistic":{"features":STATIC_FEATURES,"max_iter":2000,"class_weight":"balanced","random_state":42},
                                   "temporal_xgboost":{"features":FEATURES,"n_estimators":250,"max_depth":4,"learning_rate":0.05,
                                                       "subsample":0.85,"colsample_bytree":0.85,"reg_lambda":1.0,"random_state":42,
                                                       "n_jobs":4,"tree_method":"hist"},
                                   "calibration":{"method":"evaluation only; no probability calibration transform applied","bins":10}},
             "features":FEATURES, "next_segment":segment_metrics, "churn_90d":churn_metrics,
             "limitations":["RFM segment names are heuristic, relative quintile labels.","A 90-day inactive outcome is a proxy label, not observed contractual churn.","The dataset ends on 2011-12-09; test periods cannot represent current customer behaviour.","Calibration scores measure probability reliability; predictions are not post-hoc calibrated.","No result establishes that temporal features outperform static RFM unless the reported metrics show it."]}
    (OUTPUT_DIR/"model_metrics.json").write_text(json.dumps(metrics,indent=2,default=str))
    # Refit on all valid historical labels and score each customer in the latest observable snapshot.
    latest=samples[samples.Month==samples.Month.max()].copy()
    full_encoder=LabelEncoder().fit(next_valid.NextSegment.astype(str))
    full_y=full_encoder.transform(next_valid.NextSegment.astype(str))
    full_model=_xgb_params(len(full_encoder.classes_)); full_model.fit(next_valid[FEATURES].fillna(0),full_y)
    latest_x=latest[FEATURES].fillna(0)
    predicted=full_encoder.inverse_transform(full_model.predict(latest_x).astype(int))
    predictions=latest[["Customer ID","Month","Segment","RecencyDays" if "RecencyDays" in latest else "R_12m","F_12m","M_12m"]].copy()
    predictions["PredictedNextSegment"]=predicted
    predictions["PredictedNextSegmentProbability"]=full_model.predict_proba(latest_x).max(axis=1)
    try:
        shap_raw, _ = _shap_values(full_model, latest_x)
        if shap_raw.ndim == 3:
            local = np.stack([shap_raw[i, :, int(class_id)] for i, class_id in enumerate(full_model.predict(latest_x).astype(int))])
        else:
            local = shap_raw
        predictions["SHAPContributionsNextSegment"] = [json.dumps({f:float(v) for f,v in zip(FEATURES,row)}) for row in local]
        _save_global_shap(full_model, latest_x, "next_segment")
    except Exception as exc:
        metrics["next_segment_shap_status"] = f"Unavailable: {exc}"
    if churn_metrics.get("status")=="evaluated":
        all_churn=samples.dropna(subset=["Churn90d"])
        churn_full=_xgb_params(2,binary=True); churn_full.fit(all_churn[FEATURES].fillna(0),all_churn.Churn90d.astype(int))
        predictions["ChurnProbability90d"]=churn_full.predict_proba(latest_x)[:,1]
        churn_full.save_model(str(MODEL_DIR/"churn_xgboost.json"))
        try:
            raw, contributions=_shap_values(churn_full,latest_x)
            predictions["SHAPContributions"]= [json.dumps({f:float(v) for f,v in zip(FEATURES,row)}) for row in contributions]
            _save_global_shap(churn_full,latest_x,"churn")
        except Exception as exc:
            metrics["shap_status"] = f"Unavailable: {exc}"
    full_model.save_model(str(MODEL_DIR/"next_segment_xgboost.json"))
    joblib.dump(full_encoder,MODEL_DIR/"next_segment_label_encoder.joblib")
    predictions.to_csv(OUTPUT_DIR/"current_predictions.csv",index=False)
    (OUTPUT_DIR/"model_metrics.json").write_text(json.dumps(metrics,indent=2,default=str))
    print(json.dumps(metrics,indent=2,default=str))

if __name__=="__main__": main()
