from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import json
import pandas as pd
from pathlib import Path
from uuid import uuid4
from app.services import data_service
from app.services.data_service import rfm, summary
from app.services.report_service import sales_report
from app.services.temporal_service import journeys
from app.services.recommendation_service import recommend
from app.config import OUTPUT_DIR

app = FastAPI(title="PulseIQ API", version="0.1.0", description="Local customer analytics prototype. Predictions remain unavailable until validated models are trained.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["GET", "POST"], allow_headers=["*"])

@app.get("/api/health")
def health(): return {"status": "ok"}

@app.get("/api/summary")
def get_summary(): return summary()

@app.post("/api/dataset/upload")
async def upload_dataset(request: Request, filename: str = Query(..., min_length=1, max_length=255)):
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xlsm"}:
        raise HTTPException(status_code=415, detail="Upload a .csv, .xlsx, or .xlsm sales file.")
    chunks = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > 100 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="The upload exceeds the 100 MB limit.")
        chunks.append(chunk)
    body = b"".join(chunks)
    if not body:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    safe_stem = "".join(ch for ch in Path(filename).stem if ch.isalnum() or ch in "-_ ").strip().replace(" ", "_")[:60] or "sales_data"
    data_service.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = data_service.UPLOAD_DIR / f"{safe_stem}_{uuid4().hex[:10]}{suffix}"
    try:
        with destination.open("xb") as stream:
            stream.write(body)
        frame = data_service.load_transactions(destination)
        purchases = frame[(frame["Quantity"] > 0) & (frame["Price"] > 0) & (~frame["Invoice"].astype(str).str.startswith("C"))]
        if purchases.empty:
            raise HTTPException(status_code=422, detail="No positive purchase rows were found after cleaning.")
        data_service.set_active_dataset(destination)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Could not activate the sales file: {exc}") from exc
    return {"ok": True, "dataset": destination.name, "source_rows": int(len(frame)), "purchase_rows": int(len(purchases)), "report_url": "/api/report"}

@app.get("/api/report")
def get_sales_report():
    try:
        return sales_report()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.get("/api/report/download")
def download_sales_report():
    try:
        report = sales_report()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    safe_name = Path(report["dataset"]).stem.replace('"', "")
    return JSONResponse(report, headers={"Content-Disposition": f'attachment; filename="pulseiq_{safe_name}_report.json"'})

@app.get("/api/customers")
def customers(search: str = "", segment: str = "", country: str = "", activity_period_days: int | None = Query(None, ge=1, le=3650), has_prediction: bool = False, limit: int = Query(100, ge=1, le=10000)):
    df = rfm()
    if search: df = df[df["Customer ID"].astype(str).str.contains(search, regex=False)]
    if segment: df = df[df["Segment"] == segment]
    if country: df = df[df["Country"].astype(str) == country]
    if activity_period_days is not None: df = df[df["RecencyDays"] <= activity_period_days]
    if has_prediction:
        prediction_path = OUTPUT_DIR / "current_predictions.csv"
        if data_service.is_original_dataset() and prediction_path.exists():
            try:
                prediction_ids = pd.read_csv(prediction_path, usecols=["Customer ID"])["Customer ID"].dropna().unique()
                df = df[df["Customer ID"].isin(prediction_ids)]
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Could not read prediction customer IDs: {exc}") from exc
        else:
            df = df.iloc[0:0]
    df = df.sort_values(["RecencyDays", "Monetary"], ascending=[False, False])
    cols = ["Customer ID", "Country", "RecencyDays", "Frequency", "Monetary", "R_Score", "F_Score", "M_Score", "Segment", "InterPurchaseDays", "TenureDays", "AvgBasketValue"]
    page = df[cols].head(limit).astype(object).where(pd.notna(df[cols].head(limit)), None)
    return {"items": page.to_dict("records"), "total": int(len(df))}

@app.get("/api/filters")
def filters():
    df = rfm()
    return {"segments": sorted(df["Segment"].dropna().unique().tolist()), "countries": sorted(df["Country"].dropna().astype(str).unique().tolist())}

@app.get("/api/segments")
def segments():
    df=rfm()
    if df.empty: return {"segments": [], "customers": []}
    grouped=df.groupby("Segment", observed=False).agg(customers=("Customer ID", "count"), avg_spend=("Monetary", "mean"), avg_frequency=("Frequency", "mean")).reset_index()
    grouped["share"] = grouped["customers"] / max(1, len(df))
    return {"segments": grouped.where(grouped.notna(), None).to_dict("records"), "customers": int(len(df))}

@app.get("/api/journeys")
def get_journeys(customer_id: int | None = None): return journeys(customer_id)

@app.get("/api/model-evaluation")
def model_evaluation():
    if not data_service.is_original_dataset():
        return {"available": False, "message": "Evaluation artifacts belong to the original training dataset and are not applicable to this upload."}
    path = OUTPUT_DIR / "model_metrics.json"
    return json.loads(path.read_text()) if path.exists() else {"available": False, "message": "No chronological model evaluation has been run."}

@app.get("/api/model-evaluation/download")
def download_model_evaluation():
    path = OUTPUT_DIR / "model_metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No model evaluation report is available to download.")
    return FileResponse(path, media_type="application/json", filename="pulseiq_model_evaluation.json")

@app.get("/api/explainability/global")
def global_explanations():
    if not data_service.is_original_dataset():
        return {"available": False, "message": "Global SHAP artifacts were produced from the original training dataset and are not applied to this upload."}
    paths = {"next_segment": OUTPUT_DIR / "next_segment_shap_importance.json", "inactivity": OUTPUT_DIR / "churn_shap_importance.json"}
    results = {}
    for target, path in paths.items():
        if path.exists():
            try:
                results[target] = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise HTTPException(status_code=500, detail=f"Could not read {target} SHAP importance: {exc}") from exc
        else:
            results[target] = []
    return {"available": any(results.values()), **results}

@app.get("/api/predictions/export.csv")
def download_predictions():
    if not data_service.is_original_dataset():
        raise HTTPException(status_code=409, detail="Prediction outputs belong to the original training dataset and are unavailable for this upload.")
    path = OUTPUT_DIR / "current_predictions.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No customer prediction export is available.")
    return FileResponse(path, media_type="text/csv", filename="pulseiq_customer_predictions.csv")

@app.get("/api/predictions/queue")
def prediction_queue(search: str = "", limit: int = Query(200, ge=1, le=1000)):
    """Return a compact, searchable customer-level view of current model outputs."""
    if not data_service.is_original_dataset():
        return {"available": False, "items": [], "message": "Prediction outputs are only available for the original training dataset."}
    path = OUTPUT_DIR / "current_predictions.csv"
    if not path.exists():
        return {"available": False, "items": [], "message": "No customer prediction export is available."}
    try:
        predictions = pd.read_csv(path, usecols=["Customer ID", "Month", "Segment", "PredictedNextSegment", "PredictedNextSegmentProbability", "ChurnProbability90d"])
        if search:
            predictions = predictions[predictions["Customer ID"].astype(str).str.contains(search, regex=False)]
        total = len(predictions)
        predictions = predictions.sort_values(["ChurnProbability90d", "PredictedNextSegmentProbability"], ascending=[False, False], na_position="last").head(limit)
        items = []
        for row in predictions.to_dict("records"):
            items.append({key: (None if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)) else value) for key, value in row.items()})
        return {"available": True, "items": items, "returned": len(items), "total": total, "limit": limit}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not build the prediction queue: {exc}") from exc

@app.get("/api/predictions/{customer_id}")
def prediction(customer_id: int):
    df=rfm(); row=df[df["Customer ID"] == customer_id]
    if not data_service.is_original_dataset():
        return {"available": False, "customer_id": customer_id, "message": "Prediction models were trained on the original Online Retail II dataset and are not applied to this upload.", "profile": row.iloc[0].where(row.iloc[0].notna(), None).to_dict() if not row.empty else None}
    prediction_path = OUTPUT_DIR / "current_predictions.csv"
    if prediction_path.exists():
        outputs = pd.read_csv(prediction_path)
        found = outputs[outputs["Customer ID"] == customer_id]
        if not found.empty:
            record = found.iloc[0].where(found.iloc[0].notna(), None).to_dict()
            for key in ["SHAPContributionsNextSegment", "SHAPContributions"]:
                if record.get(key): record[key] = json.loads(record[key])
            return {"available": True, "customer_id": customer_id, "prediction": record, "profile": row.iloc[0].where(row.iloc[0].notna(), None).to_dict() if not row.empty else None}
    if prediction_path.exists():
        message = "This customer has no saved prediction in the latest scored period. Choose a customer with predictions for local SHAP values; global feature importance may still be available."
    else:
        message = "No validated customer prediction artifact is available yet."
    return {"available": False, "customer_id": customer_id, "message": message, "profile": row.iloc[0].where(row.iloc[0].notna(), None).to_dict() if not row.empty else None}

@app.get("/api/recommendations/{customer_id}")
def recommendation(customer_id: int):
    df=rfm(); row=df[df["Customer ID"] == customer_id]
    if row.empty: return {"available": False, "message": "Customer not found in cleaned purchase data."}
    record=row.iloc[0].where(row.iloc[0].notna(), None).to_dict()
    prediction_result = prediction(customer_id)
    model_output = prediction_result.get("prediction") if prediction_result.get("available") else None
    return {"available": True, **recommend(record, model_output)}
