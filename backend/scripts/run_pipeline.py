"""Prepare actual Online Retail II RFM outputs; does not train predictive models."""
import json, os
from pathlib import Path
import pandas as pd
from app.ml.segmentation import segment_customers

ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
path = Path(os.getenv("PULSEIQ_DATA_PATH", ROOT / "data/raw/online_retail_II.xlsx")).expanduser()
out_dir = ROOT / "data/processed"
summary_dir = ROOT / "outputs"
required = {"Invoice", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"}

def main():
    if not path.exists(): raise SystemExit(f"Dataset not found: {path}. Set PULSEIQ_DATA_PATH.")
    book = pd.ExcelFile(path)
    frames = [pd.read_excel(path, sheet_name=s) for s in book.sheet_names]
    raw = pd.concat(frames, ignore_index=True)
    missing = required - set(raw.columns)
    if missing: raise SystemExit(f"Missing required columns: {sorted(missing)}")
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], errors="coerce")
    raw["Customer ID"] = pd.to_numeric(raw["Customer ID"], errors="coerce")
    raw["Quantity"] = pd.to_numeric(raw["Quantity"], errors="coerce")
    raw["Price"] = pd.to_numeric(raw["Price"], errors="coerce")
    valid = raw.dropna(subset=["InvoiceDate", "Customer ID", "Quantity", "Price", "Country"]).copy()
    purchases = valid[(valid.Quantity > 0) & (valid.Price > 0) & (~valid.Invoice.astype(str).str.startswith("C"))].copy()
    purchases["Revenue"] = purchases.Quantity * purchases.Price
    as_of = purchases.InvoiceDate.max().normalize() + pd.Timedelta(days=1)
    invoice = purchases.groupby(["Customer ID", "Invoice"], as_index=False).agg(InvoiceDate=("InvoiceDate", "max"), Monetary=("Revenue", "sum"), Country=("Country", "last"))
    profile = invoice.groupby("Customer ID").agg(LastPurchase=("InvoiceDate", "max"), FirstPurchase=("InvoiceDate", "min"), Frequency=("Invoice", "nunique"), Monetary=("Monetary", "sum"), Country=("Country", "last"), InvoiceDates=("InvoiceDate", lambda s: sorted(s.tolist()))).reset_index()
    profile["RecencyDays"] = (as_of - profile.LastPurchase).dt.days
    profile["TenureDays"] = (profile.LastPurchase - profile.FirstPurchase).dt.days
    profile["AvgBasketValue"] = profile.Monetary / profile.Frequency
    profile["InterPurchaseDays"] = profile.InvoiceDates.map(lambda ds: float(pd.Series(ds).diff().dt.total_seconds().div(86400).mean()) if len(ds) > 1 else None)
    profile = segment_customers(profile).drop(columns=["InvoiceDates"])
    profile.to_csv(out_dir / "customer_rfm.csv", index=False)
    snapshot = {"available": True, "dataset": path.name, "sheets": book.sheet_names, "source_rows": int(len(raw)), "valid_customer_rows": int(len(valid)), "purchase_rows": int(len(purchases)), "excluded_after_required_fields": int(len(raw)-len(valid)), "excluded_non_purchase_rows": int(len(valid)-len(purchases)), "customers": int(len(profile)), "repeat_customers": int((profile.Frequency > 1).sum()), "at_risk_customers": int(profile.Segment.isin(["Needs Attention", "At Risk"]).sum()), "date_min": purchases.InvoiceDate.min().date().isoformat(), "date_max": purchases.InvoiceDate.max().date().isoformat(), "segment_counts": {str(k): int(v) for k,v in profile.Segment.value_counts().items()}, "model_status": "Training failed once; retry needed", "training_attempts": 1, "prediction_available": False, "shap_available": False, "definitions": {"purchase_rows": "Quantity > 0, Price > 0, and invoice number does not start with C", "return_rows": "Negative-quantity rows whose invoice number does not start with C; kept separate from cancellations.", "cancelled_invoices": "Distinct invoice numbers beginning with C.", "repeat_customer": "More than one distinct purchase invoice", "at_risk_customer": "Relative RFM segment is Needs Attention or At Risk", "segment_method": "Relative per-snapshot quintile scores; heuristic labels, not validated business definitions.", "recommendation_priority": "Review for current/predicted Needs Attention/At Risk or predicted 90-day inactivity probability >= 0.5; heuristic and not calibrated."}}
    (summary_dir / "data_summary.json").write_text(json.dumps(snapshot, indent=2))
    # Build a real, static deep-dive snapshot so the dashboard remains useful when
    # the local API is still starting or cannot be installed in an offline environment.
    monthly = purchases.assign(Month=purchases.InvoiceDate.dt.to_period("M")).groupby("Month").agg(
        revenue=("Revenue", "sum"), orders=("Invoice", "nunique"), customers=("Customer ID", "nunique"), units=("Quantity", "sum")
    ).reset_index()
    monthly["Month"] = monthly["Month"].astype(str)
    countries = purchases.groupby("Country").agg(
        revenue=("Revenue", "sum"), orders=("Invoice", "nunique"), customers=("Customer ID", "nunique"), units=("Quantity", "sum")
    ).reset_index().sort_values("revenue", ascending=False).head(12)
    products = purchases.groupby(["StockCode", "Description"], dropna=False).agg(
        revenue=("Revenue", "sum"), orders=("Invoice", "nunique"), customers=("Customer ID", "nunique"), units=("Quantity", "sum")
    ).reset_index().sort_values("revenue", ascending=False).head(12)
    cancellation_rows = valid[valid.Invoice.astype(str).str.startswith("C")]
    return_rows = valid[(valid.Quantity < 0) & (~valid.Invoice.astype(str).str.startswith("C"))]
    invoice_count = int(invoice["Invoice"].nunique())
    top_customers = profile.sort_values("Monetary", ascending=False).head(20).copy()
    top_at_risk = profile[profile.Segment.isin(["Needs Attention", "At Risk"])].sort_values("RecencyDays", ascending=False).head(20)
    customers_for_journeys = pd.concat([top_at_risk, top_customers]).drop_duplicates("Customer ID")["Customer ID"]
    temporal_states = []
    months = pd.period_range(invoice.InvoiceDate.min().to_period("M"), invoice.InvoiceDate.max().to_period("M"), freq="M")
    for month in months:
        cutoff = (month + 1).start_time
        history = invoice[(invoice.InvoiceDate >= cutoff - pd.DateOffset(months=12)) & (invoice.InvoiceDate < cutoff)]
        if history.empty:
            continue
        state = history.groupby("Customer ID").agg(
            LastPurchase=("InvoiceDate", "max"), Frequency=("Invoice", "nunique"), Monetary=("Monetary", "sum"), Country=("Country", "last")
        ).reset_index()
        state["RecencyDays"] = (cutoff - state.LastPurchase).dt.days
        state["Month"] = month
        temporal_states.append(segment_customers(state))
    states = pd.concat(temporal_states, ignore_index=True) if temporal_states else pd.DataFrame()
    journey_periods, transitions, journey_customers = [], [], []
    if not states.empty:
        period_counts = states.groupby(["Month", "Segment"]).size().rename("customers").reset_index()
        period_counts["Month"] = period_counts["Month"].astype(str)
        journey_periods = json.loads(period_counts.to_json(orient="records"))
        ordered = states.sort_values(["Customer ID", "Month"]).copy()
        ordered["To"] = ordered.groupby("Customer ID")["Segment"].shift(-1)
        ordered["NextMonth"] = ordered.groupby("Customer ID")["Month"].shift(-1)
        valid_pairs = ordered[ordered.NextMonth == ordered.Month + 1]
        transition_counts = valid_pairs.groupby(["Segment", "To"]).size().rename("count").reset_index().rename(columns={"Segment":"From"})
        transitions = json.loads(transition_counts.to_json(orient="records"))
        journeys = states[states["Customer ID"].isin(customers_for_journeys)][["Customer ID", "Month", "Segment", "RecencyDays", "Frequency", "Monetary"]].copy()
        journeys["Month"] = journeys["Month"].astype(str)
        journey_customers = json.loads(journeys.to_json(orient="records"))
    def json_records(frame):
        return json.loads(frame.to_json(orient="records"))
    snapshot["deep_dive"] = {
        "orders": invoice_count,
        "gross_sales": float(purchases.Revenue.sum()),
        "average_order_value": float(purchases.Revenue.sum() / invoice_count) if invoice_count else 0,
        "average_units_per_order": float(purchases.Quantity.sum() / invoice_count) if invoice_count else 0,
        "unique_products": int(purchases.StockCode.nunique()) if "StockCode" in purchases else None,
        "cancellation_invoice_count": int(cancellation_rows.Invoice.nunique()),
        "return_row_count": int(len(return_rows)),
        "returned_value_abs": float((return_rows.Quantity * return_rows.Price).abs().sum()),
        "monthly": json_records(monthly),
        "countries": json_records(countries),
        "products": json_records(products),
        "journeys": {"periods": journey_periods, "transitions": transitions, "customers": journey_customers},
        "all_customers": json_records(profile[["Customer ID", "Country", "RecencyDays", "Frequency", "Monetary", "R_Score", "F_Score", "M_Score", "Segment", "LastPurchase", "TenureDays", "AvgBasketValue", "InterPurchaseDays"]]),
        "top_customers": json_records(top_customers[["Customer ID", "Country", "RecencyDays", "Frequency", "Monetary", "R_Score", "F_Score", "M_Score", "Segment", "LastPurchase", "TenureDays", "AvgBasketValue", "InterPurchaseDays"]])
    }
    analytics_path = summary_dir / "dashboard_snapshot.json"
    analytics_path.write_text(json.dumps(snapshot, indent=2))
    public_dir = ROOT / "frontend" / "public"
    public_dir.mkdir(parents=True, exist_ok=True)
    (public_dir / "analytics_snapshot.json").write_text(json.dumps(snapshot, separators=(",", ":")))
    print(json.dumps({"dataset": path.name, "source_rows": snapshot["source_rows"], "purchase_rows": snapshot["purchase_rows"], "customers": snapshot["customers"], "snapshot": str(public_dir / "analytics_snapshot.json")}))

if __name__ == "__main__": main()
