"""Evidence-linked deterministic retention fallback."""
import json
import os
from urllib.request import Request, urlopen

FEATURE_LABELS = {
    "R_3m": "recency over 3 months", "F_3m": "purchase frequency over 3 months", "M_3m": "spend over 3 months",
    "R_6m": "recency over 6 months", "F_6m": "purchase frequency over 6 months", "M_6m": "spend over 6 months",
    "R_12m": "recency over 12 months", "F_12m": "purchase frequency over 12 months", "M_12m": "spend over 12 months",
    "Frequency_Change_3m": "recent purchase-frequency change", "Monetary_Change_3m": "recent spending change",
    "InterPurchaseDays_12m": "average time between purchases", "TenureDays_12m": "customer tenure"
}

def _llm_suggestions(evidence_payload: dict) -> list[str] | None:
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    endpoint = os.getenv("LLM_API_BASE", "https://api.openai.com/v1/chat/completions")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    body = {"model": model, "temperature": 0.2, "messages": [
        {"role":"system","content":"You write brief customer retention actions. Use only supplied observed facts and model outputs. Do not claim to predict churn yourself, infer causality, invent products or customer history, or add unsupported details. Return only JSON: {\"recommendations\":[string,string],\"rationale\":string}."},
        {"role":"user","content":json.dumps(evidence_payload, separators=(",",":"))}
    ]}
    request = Request(endpoint, data=json.dumps(body).encode("utf-8"), headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"}, method="POST")
    try:
        with urlopen(request, timeout=25) as response:
            result=json.loads(response.read().decode("utf-8"))
        content=result["choices"][0]["message"]["content"]
        parsed=json.loads(content)
        suggestions=parsed.get("recommendations")
        return [str(x) for x in suggestions[:3] if isinstance(x,str)] if isinstance(suggestions,list) else None
    except Exception:
        return None

def recommend(customer: dict, model_output: dict | None = None) -> dict:
    """Create transparent rules from observed profile and optional actual model evidence."""
    segment = customer.get("Segment", "Unknown")
    recency = int(customer.get("RecencyDays", 0) or 0)
    frequency = int(customer.get("Frequency", 0) or 0)
    reasons, actions = [], []
    if segment in {"At Risk", "Needs Attention"}:
        reasons.append(f"The current relative RFM segment is {segment}.")
        actions.append("Send a targeted re-engagement message with a relevant product category based on prior purchases.")
    if recency >= 90:
        reasons.append(f"The last observed purchase was {recency} days before the dataset cut-off.")
        actions.append("Test a return offer and measure response against a holdout group.")
    if frequency <= 1:
        reasons.append("Only one distinct purchase invoice is present in the observed history.")
        actions.append("Follow up with a second-purchase recommendation using the customer's purchased product category.")
    result = {"provider": "Deterministic rule-based fallback", "recommendations": list(dict.fromkeys(actions)), "evidence": reasons,
              "disclaimer": "Suggestions use observed history and available model evidence. They do not establish causality."}
    if model_output:
        result["predicted_next_segment"] = model_output.get("PredictedNextSegment")
        result["next_segment_probability"] = model_output.get("PredictedNextSegmentProbability")
        result["churn_probability_90d"] = model_output.get("ChurnProbability90d")
        if model_output.get("PredictedNextSegment") in {"At Risk", "Needs Attention"}:
            result["evidence"].append(f"The trained model predicts the next segment as {model_output['PredictedNextSegment']}.")
            result["recommendations"].append("Prioritize a retention contact before the predicted segment transition and compare with a holdout group.")
        for col, title in [("SHAPContributionsNextSegment", "next-segment"), ("SHAPContributions", "churn")]:
            raw = model_output.get(col)
            try:
                values = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(values, dict) and values:
                    top = sorted(values.items(), key=lambda item: abs(float(item[1])), reverse=True)[:3]
                    descriptions = [FEATURE_LABELS.get(k, k.replace("_", " ")) for k,v in top]
                    result["evidence"].append(f"SHAP for the {title} prediction highlighted " + ", ".join(descriptions) + ".")
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
    if not result["recommendations"]:
        result["evidence"].append(f"The current relative RFM segment is {segment}; no specific deterioration rule was triggered.")
        result["recommendations"].append("Review recent purchase history before selecting a retention offer.")
    result["recommendations"] = list(dict.fromkeys(result["recommendations"]))
    predicted_segment = model_output.get("PredictedNextSegment") if model_output else None
    result["priority"] = "Review" if segment in {"At Risk", "Needs Attention"} or predicted_segment in {"At Risk", "Needs Attention"} or (model_output and float(model_output.get("ChurnProbability90d") or 0) >= 0.5) else "Standard"
    payload={"observed_profile":{"segment":segment,"recency_days":recency,"invoice_frequency":frequency,"monetary_value":customer.get("Monetary")},
             "observed_evidence":result["evidence"],"model_outputs":{k:model_output.get(k) for k in ["PredictedNextSegment","PredictedNextSegmentProbability","ChurnProbability90d"] if model_output and model_output.get(k) is not None}}
    suggestions=_llm_suggestions(payload)
    if suggestions:
        result["recommendations"]=suggestions
        result["provider"]="LLM grounded in supplied evidence"
        result["disclaimer"]="The LLM drafts actions only. Predictions and behavioural evidence come from the supplied model outputs and observed data."
    return result
