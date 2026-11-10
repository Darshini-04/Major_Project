# PulseIQ architecture

## Current request flow

```text
Online Retail II workbook (local XLSX)
        ↓
Pandas ingestion and schema checks
        ↓
Cleaning and positive-purchase filtering
        ↓
Invoice aggregation → customer RFM and behavioural profile
        ↓
Relative RFM segmentation (heuristic)
        ↓
FastAPI JSON endpoints
        ↓
Next.js dashboard and deterministic retention fallback
```

The workbook path is configured through `PULSEIQ_DATA_PATH`; the raw workbook is not stored in Git. The API caches one in-memory read per process. There is no database, authentication, cloud deployment, or background worker.

## Planned research flow

Transaction snapshots across chronological observation windows → time-consistent segmentation → next-period segment and future inactivity labels → temporal train/test evaluation against a static RFM baseline → SHAP explanations for trained models → evidence-grounded recommendations. The planned stages remain pending until implemented and evaluated. Current segment labels are relative quintiles and should not be interpreted as business truth or a churn label.
