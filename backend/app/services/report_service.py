import json
import pandas as pd
from app.services import data_service
from app.services.data_service import transactions, purchase_rows, rfm
from app.services.temporal_service import journeys


def _records(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def sales_report() -> dict:
    raw = transactions()
    purchases = purchase_rows()
    if purchases.empty:
        raise ValueError("No positive purchase rows remain after cleaning; check Quantity, Price, and Invoice values.")
    purchases = purchases.copy()
    purchases["Revenue"] = purchases["Quantity"] * purchases["Price"]
    purchases["Month"] = purchases["InvoiceDate"].dt.to_period("M")
    invoice_count = int(purchases["Invoice"].nunique())
    monthly = purchases.groupby("Month").agg(revenue=("Revenue", "sum"), orders=("Invoice", "nunique"), customers=("Customer ID", "nunique"), units=("Quantity", "sum")).reset_index()
    monthly["Month"] = monthly["Month"].astype(str)
    countries = purchases.groupby("Country").agg(revenue=("Revenue", "sum"), orders=("Invoice", "nunique"), customers=("Customer ID", "nunique"), units=("Quantity", "sum")).reset_index().sort_values("revenue", ascending=False).head(12)
    stock = purchases["StockCode"].astype(str) if "StockCode" in purchases else pd.Series("Unspecified", index=purchases.index)
    desc = purchases["Description"].fillna(stock).astype(str) if "Description" in purchases else stock
    product_frame = purchases.assign(_stock=stock, _description=desc).groupby(["_stock", "_description"], dropna=False).agg(revenue=("Revenue", "sum"), orders=("Invoice", "nunique"), customers=("Customer ID", "nunique"), units=("Quantity", "sum")).reset_index().sort_values("revenue", ascending=False).head(12).rename(columns={"_stock": "StockCode", "_description": "Description"})
    cancellations = raw[raw["Invoice"].astype(str).str.startswith("C")]
    returns = raw[(raw["Quantity"] < 0) & (~raw["Invoice"].astype(str).str.startswith("C"))]
    profiles = rfm()
    currency = None
    if "Currency" in raw:
        currencies = raw["Currency"].dropna().astype(str).unique().tolist()
        if len(currencies) == 1 and len(currencies[0]) == 3:
            currency = currencies[0].upper()
    return {
        "dataset": data_service.DATA_PATH.name,
        "currency": currency,
        "source_rows": int(raw.attrs.get("source_rows", len(raw))), "valid_rows": int(len(raw)), "purchase_rows": int(len(purchases)),
        "customers": int(profiles["Customer ID"].nunique()) if not profiles.empty else 0,
        "date_min": purchases["InvoiceDate"].min().date().isoformat(), "date_max": purchases["InvoiceDate"].max().date().isoformat(),
        "orders": invoice_count, "gross_sales": float(purchases["Revenue"].sum()),
        "average_order_value": float(purchases["Revenue"].sum() / invoice_count) if invoice_count else 0,
        "average_units_per_order": float(purchases["Quantity"].sum() / invoice_count) if invoice_count else 0,
        "unique_products": int(purchases["StockCode"].nunique()) if "StockCode" in purchases else None,
        "cancellation_invoice_count": int(cancellations["Invoice"].nunique()), "return_row_count": int(len(returns)),
        "returned_value_abs": float((returns["Quantity"] * returns["Price"]).abs().sum()),
        "monthly": _records(monthly), "countries": _records(countries),
        "products": [{"StockCode": row["StockCode"], "Description": row["Description"], **{k: row[k] for k in ("revenue", "orders", "customers", "units")}} for _, row in product_frame.iterrows()],
        "journeys": journeys(),
        "all_customers": _records(profiles[["Customer ID", "Country", "RecencyDays", "Frequency", "Monetary", "R_Score", "F_Score", "M_Score", "Segment", "LastPurchase", "TenureDays", "AvgBasketValue", "InterPurchaseDays"]]) if not profiles.empty else [],
        "definitions": {"purchase_rows": "Quantity > 0, Price > 0, and invoice number does not start with C", "returns": "Negative-quantity rows excluding cancellation invoices; reported separately", "gross_sales": "Sum of Quantity × Price across positive purchase rows", "average_order_value": "Gross sales divided by distinct purchase invoices", "customer_segments": "Relative RFM quintile scores computed from the active dataset"},
    }
