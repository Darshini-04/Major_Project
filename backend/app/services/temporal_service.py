import pandas as pd
from app.services.data_service import purchase_rows, rfm
from functools import lru_cache


@lru_cache(maxsize=1)
def monthly_segments() -> pd.DataFrame:
    df = purchase_rows()
    if df.empty:
        return pd.DataFrame()
    months = pd.period_range(df["InvoiceDate"].min().to_period("M"), df["InvoiceDate"].max().to_period("M"), freq="M")
    snapshots = []
    for month in months:
        end = month.end_time.normalize() + pd.Timedelta(days=1)
        start = (month - 11).start_time
        history = df[(df["InvoiceDate"] >= start) & (df["InvoiceDate"] < end)]
        if history.empty:
            continue
        inv = history.groupby(["Customer ID", "Invoice"], as_index=False).agg(InvoiceDate=("InvoiceDate", "max"), Monetary=("Revenue", "sum"), Country=("Country", "last"))
        prof = inv.groupby("Customer ID").agg(LastPurchase=("InvoiceDate", "max"), Frequency=("Invoice", "nunique"), Monetary=("Monetary", "sum"), Country=("Country", "last")).reset_index()
        prof["RecencyDays"] = (end - prof["LastPurchase"]).dt.days
        prof["Month"] = str(month)
        from app.ml.segmentation import segment_customers
        snapshots.append(segment_customers(prof))
    return pd.concat(snapshots, ignore_index=True) if snapshots else pd.DataFrame()


def journeys(customer_id: int | None = None) -> dict:
    states = monthly_segments()
    if states.empty: return {"periods": [], "transitions": [], "customers": []}
    ordered = states.sort_values(["Customer ID", "Month"])
    following = ordered.groupby("Customer ID")["Segment"].shift(-1)
    transitions = pd.DataFrame({"From": ordered["Segment"], "To": following}).dropna()
    matrix = transitions.groupby(["From", "To"]).size().rename("count").reset_index().to_dict("records") if not transitions.empty else []
    selected = states[states["Customer ID"] == customer_id] if customer_id is not None else states
    if customer_id is None: selected = selected.sort_values("Month").groupby("Month").head(1)
    return {"periods": states.groupby(["Month", "Segment"]).size().rename("customers").reset_index().to_dict("records"), "transitions": matrix,
            "customers": selected[["Customer ID", "Month", "Segment", "RecencyDays", "Frequency", "Monetary"]].to_dict("records")}
