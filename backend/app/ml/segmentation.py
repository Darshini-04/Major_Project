"""Transparent relative RFM segmentation shared by historical snapshots."""
import numpy as np
import pandas as pd

SEGMENTS = ["Champions", "Loyal", "Potential Loyalists", "Needs Attention", "At Risk"]


def _quintile_score(values: pd.Series, *, lower_is_better: bool = False) -> pd.Series:
    # Rank first so ties remain deterministic, then map percentile ranks to 1..5.
    ranked = values.rank(method="first", ascending=True, pct=True)
    score = np.ceil(ranked * 5).clip(1, 5).astype(int)
    return 6 - score if lower_is_better else score


def segment_customers(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Assign relative quintile RFM labels; higher R/F/M scores indicate higher value."""
    result = snapshot.copy()
    if result.empty:
        result["R_Score"] = pd.Series(dtype=int)
        result["F_Score"] = pd.Series(dtype=int)
        result["M_Score"] = pd.Series(dtype=int)
        result["Segment"] = pd.Series(dtype=str)
        return result
    result["R_Score"] = _quintile_score(result["RecencyDays"], lower_is_better=True)
    result["F_Score"] = _quintile_score(result["Frequency"])
    result["M_Score"] = _quintile_score(result["Monetary"])
    score = result[["R_Score", "F_Score", "M_Score"]].mean(axis=1)
    result["Segment"] = pd.cut(score, bins=[0, 1.8, 2.6, 3.4, 4.2, 5.01], labels=list(reversed(SEGMENTS))).astype(str)
    return result
