from functools import lru_cache
import hashlib
from pathlib import Path
import pandas as pd
import json
from fastapi import HTTPException
from app.config import DATA_PATH as DEFAULT_DATA_PATH, OUTPUT_DIR, ROOT
from app.ml.segmentation import segment_customers

REQUIRED = {"Invoice", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"}
UPLOAD_DIR = ROOT / "data" / "uploads"
ACTIVE_DATASET_FILE = UPLOAD_DIR / "active_dataset.json"

def _initial_dataset_path():
    try:
        selected = Path(json.loads(ACTIVE_DATASET_FILE.read_text())["path"]).resolve()
        if selected.is_relative_to(UPLOAD_DIR.resolve()) and selected.is_file():
            return selected
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return DEFAULT_DATA_PATH

DATA_PATH = _initial_dataset_path()

def set_active_dataset(path: Path) -> None:
    """Select a validated dataset without touching the source workbook or snapshot."""
    global DATA_PATH
    DATA_PATH = path.resolve()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_DATASET_FILE.write_text(json.dumps({"path": str(DATA_PATH)}, indent=2))
    transactions.cache_clear()
    rfm.cache_clear()
    from app.services.temporal_service import monthly_segments
    monthly_segments.cache_clear()

def is_original_dataset() -> bool:
    active = DATA_PATH.expanduser().resolve()
    original = DEFAULT_DATA_PATH.expanduser().resolve()
    if active == original:
        return True
    try:
        active_stat = active.stat()
        original_stat = original.stat()
    except OSError:
        return False
    return _same_dataset_content(
        str(active), active_stat.st_size, active_stat.st_mtime_ns,
        str(original), original_stat.st_size, original_stat.st_mtime_ns,
    )


@lru_cache(maxsize=8)
def _same_dataset_content(
    active_path: str, active_size: int, active_mtime_ns: int,
    original_path: str, original_size: int, original_mtime_ns: int,
) -> bool:
    """Allow saved model artifacts for an exact copy of their training workbook."""
    if active_size != original_size:
        return False

    def digest(path: str) -> str:
        value = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()

    try:
        return digest(active_path) == digest(original_path)
    except OSError:
        return False

def load_transactions(path: Path) -> pd.DataFrame:
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() in {".xlsx", ".xlsm"}:
            book = pd.ExcelFile(path)
            df = pd.concat([pd.read_excel(path, sheet_name=name) for name in book.sheet_names], ignore_index=True)
        else:
            raise ValueError("Only .csv, .xlsx, and .xlsm files are supported.")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read the sales file: {exc}") from exc
    missing = REQUIRED - set(df.columns)
    if missing:
        raise HTTPException(status_code=422, detail=f"Sales file is missing required columns: {sorted(missing)}")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Country"] = df["Country"].astype("string").str.strip()
    df["Revenue"] = df["Quantity"] * df["Price"]
    source_rows = len(df)
    df = df.dropna(subset=["InvoiceDate", "Customer ID", "Quantity", "Price", "Country"])
    df.attrs["source_rows"] = source_rows
    return df

@lru_cache(maxsize=1)
def transactions() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found at {DATA_PATH}. Set PULSEIQ_DATA_PATH or place Online Retail II at data/raw/online_retail_II.xlsx.")
    return load_transactions(DATA_PATH)


def purchase_rows() -> pd.DataFrame:
    df = transactions()
    return df[(df["Quantity"] > 0) & (df["Price"] > 0) & (~df["Invoice"].astype(str).str.startswith("C"))].copy()


@lru_cache(maxsize=1)
def rfm(as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    df = purchase_rows()
    if df.empty:
        return pd.DataFrame()
    as_of = as_of or (df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1))
    by_invoice = df.groupby(["Customer ID", "Invoice"], as_index=False).agg(InvoiceDate=("InvoiceDate", "max"), Monetary=("Revenue", "sum"), Country=("Country", "last"))
    out = by_invoice.groupby("Customer ID").agg(
        LastPurchase=("InvoiceDate", "max"), Frequency=("Invoice", "nunique"), Monetary=("Monetary", "sum"), Country=("Country", "last"),
        FirstPurchase=("InvoiceDate", "min"), InvoiceDays=("InvoiceDate", lambda s: sorted(s.tolist()))
    ).reset_index()
    out["RecencyDays"] = (as_of - out["LastPurchase"]).dt.days
    out["TenureDays"] = (out["LastPurchase"] - out["FirstPurchase"]).dt.days
    out["AvgBasketValue"] = out["Monetary"] / out["Frequency"]
    out["InterPurchaseDays"] = out["InvoiceDays"].map(lambda dates: (pd.Series(dates).diff().dt.total_seconds().div(86400).mean() if len(dates) > 1 else None))
    return segment_customers(out)


def summary() -> dict:
    raw = transactions(); clean = purchase_rows(); profile = rfm()
    counts = profile["Segment"].value_counts().to_dict() if not profile.empty else {}
    attention = int(profile["Segment"].isin(["Needs Attention", "At Risk"]).sum()) if not profile.empty else 0
    compatible = is_original_dataset()
    metric_path = OUTPUT_DIR / "model_metrics.json"
    trained = compatible and metric_path.exists()
    metric_data = json.loads(metric_path.read_text()) if trained else {}
    return {
        "dataset": DATA_PATH.name, "available": True, "source_rows": int(raw.attrs.get("source_rows", len(raw))), "valid_rows": int(len(raw)), "purchase_rows": int(len(clean)),
        "excluded_rows": int(raw.attrs.get("source_rows", len(raw)) - len(clean)), "customers": int(profile["Customer ID"].nunique()) if not profile.empty else 0,
        "repeat_customers": int((profile["Frequency"] > 1).sum()) if not profile.empty else 0,
        "at_risk_customers": attention, "segment_counts": counts,
        "date_min": clean["InvoiceDate"].min().date().isoformat() if not clean.empty else None,
        "date_max": clean["InvoiceDate"].max().date().isoformat() if not clean.empty else None,
        "model_status": "Evaluated" if trained else ("Not trained for this dataset" if not compatible else "Training failed once; retry needed"), "prediction_available": compatible and (OUTPUT_DIR / "current_predictions.csv").exists(), "shap_available": compatible and (OUTPUT_DIR / "next_segment_shap_importance.json").exists(),
        "evaluation_available": trained, "evaluation": metric_data,
        "definitions": {"purchase_rows": "Quantity > 0, Price > 0, and invoice number does not start with C", "repeat_customer": "More than one distinct purchase invoice", "at_risk_customer": "Relative RFM segment is Needs Attention or At Risk", "segment_method": "Per-snapshot quintile scores; labels are heuristic and should be validated with the research supervisor.", "recommendation_priority": "Review when the current or predicted relative segment is Needs Attention or At Risk, or a trained 90-day inactivity probability is at least 0.5; the threshold is a heuristic, not calibrated policy."}
    }
