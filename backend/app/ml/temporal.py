"""Temporal feature and label construction with strict observation/future boundaries."""
from datetime import timedelta

import pandas as pd
from app.ml.segmentation import segment_customers
from app.services.data_service import purchase_rows

FEATURES = [
    "R_3m", "F_3m", "M_3m", "Basket_3m",
    "R_6m", "F_6m", "M_6m", "Basket_6m",
    "R_12m", "F_12m", "M_12m", "Basket_12m",
    "Frequency_Change_3m", "Monetary_Change_3m", "InterPurchaseDays_12m", "TenureDays_12m",
]


def _window_features(invoice: pd.DataFrame, cutoff: pd.Timestamp, months: int, suffix: str) -> pd.DataFrame:
    start = cutoff - pd.DateOffset(months=months)
    window = invoice[(invoice["InvoiceDate"] >= start) & (invoice["InvoiceDate"] < cutoff)]
    if window.empty:
        return pd.DataFrame(columns=["Customer ID", f"R_{suffix}", f"F_{suffix}", f"M_{suffix}", f"Basket_{suffix}"])
    grouped = window.groupby("Customer ID").agg(last_purchase=("InvoiceDate", "max"), frequency=("Invoice", "nunique"), monetary=("InvoiceAmount", "sum"), first_purchase=("InvoiceDate", "min")).reset_index()
    out = grouped[["Customer ID"]].copy()
    out[f"R_{suffix}"] = (cutoff - grouped["last_purchase"]).dt.days.astype(float)
    out[f"F_{suffix}"] = grouped["frequency"].astype(float)
    out[f"M_{suffix}"] = grouped["monetary"].astype(float)
    out[f"Basket_{suffix}"] = grouped["monetary"] / grouped["frequency"].clip(lower=1)
    if suffix == "12m":
        out["TenureDays_12m"] = (grouped["last_purchase"] - grouped["first_purchase"]).dt.days.astype(float)
        intervals = window.groupby("Customer ID")["InvoiceDate"].apply(lambda dates: dates.sort_values().diff().dt.total_seconds().div(86400).mean() if dates.nunique() > 1 else float("nan"))
        out["InterPurchaseDays_12m"] = out["Customer ID"].map(intervals)
    return out


def temporal_snapshots() -> pd.DataFrame:
    """Create one customer-state row per month from only data observable by that cutoff."""
    source = purchase_rows()
    if source.empty:
        return pd.DataFrame()
    invoices = source.groupby(["Customer ID", "Invoice"], as_index=False).agg(InvoiceDate=("InvoiceDate", "max"), InvoiceAmount=("Revenue", "sum"))
    months = pd.period_range(invoices["InvoiceDate"].min().to_period("M"), invoices["InvoiceDate"].max().to_period("M"), freq="M")
    snapshots = []
    for month in months:
        cutoff = (month + 1).start_time
        features = _window_features(invoices, cutoff, 3, "3m")
        for size, name in [(6, "6m"), (12, "12m")]:
            features = features.merge(_window_features(invoices, cutoff, size, name), on="Customer ID", how="outer")
        if features.empty:
            continue
        features["Month"] = month
        features["Frequency_Change_3m"] = features["F_3m"].fillna(0) - (features["F_6m"].fillna(0) - features["F_3m"].fillna(0))
        features["Monetary_Change_3m"] = features["M_3m"].fillna(0) - (features["M_6m"].fillna(0) - features["M_3m"].fillna(0))
        for col in FEATURES:
            if col.startswith("R_"):
                features[col] = features[col].fillna(999.0)
            else:
                features[col] = features[col].fillna(0.0)
        state = features[["Customer ID", "R_12m", "F_12m", "M_12m"]].rename(columns={"R_12m":"RecencyDays", "F_12m":"Frequency", "M_12m":"Monetary"})
        features["Segment"] = segment_customers(state)["Segment"].values
        snapshots.append(features)
    return pd.concat(snapshots, ignore_index=True) if snapshots else pd.DataFrame()


def training_samples() -> pd.DataFrame:
    """Build next-month segment and observable 90-day inactivity labels."""
    states = temporal_snapshots()
    if states.empty: return states
    states = states.sort_values(["Customer ID", "Month"])
    states["NextMonth"] = states["Month"] + 1
    target = states[["Customer ID", "Month", "Segment"]].rename(columns={"Month":"NextMonth", "Segment":"NextSegment"})
    samples = states.merge(target, on=["Customer ID", "NextMonth"], how="left")
    tx = purchase_rows()
    last_observed = tx["InvoiceDate"].max()
    horizon = timedelta(days=90)
    # A churn label is valid only if the complete future horizon is in the observed dataset.
    eligible = samples["Month"].map(lambda m: (m + 1).start_time + horizon <= last_observed)
    labels = []
    customer_dates = tx.groupby("Customer ID")["InvoiceDate"].apply(lambda s: s.sort_values().to_numpy())
    for row in samples[["Customer ID", "Month"]].itertuples(index=False):
        cutoff = (row.Month + 1).start_time
        dates = customer_dates.get(row[0])
        if dates is None or cutoff + horizon > last_observed:
            labels.append(None); continue
        values = pd.to_datetime(dates)
        labels.append(int(not ((values >= cutoff) & (values < cutoff + horizon)).any()))
    samples["Churn90d"] = labels
    samples.loc[~eligible, "Churn90d"] = None
    samples["SnapshotDate"] = samples["Month"].map(lambda m: (m + 1).start_time.date().isoformat())
    return samples
