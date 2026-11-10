import unittest
from unittest.mock import patch
from app.services.recommendation_service import recommend

class RecommendationTests(unittest.TestCase):
    @patch("app.services.recommendation_service._llm_suggestions", return_value=None)
    def test_fallback_uses_observed_evidence_and_discloses_limits(self, _):
        result = recommend({"Segment":"At Risk", "RecencyDays":120, "Frequency":1, "Monetary":42})
        self.assertEqual(result["provider"], "Deterministic rule-based fallback")
        self.assertTrue(result["recommendations"])
        self.assertIn("120 days", " ".join(result["evidence"]))
        self.assertIn("causality", result["disclaimer"])

    @patch("app.services.recommendation_service._llm_suggestions", return_value=None)
    def test_model_drivers_are_connected_to_evidence(self, _):
        result = recommend({"Segment":"Loyal", "RecencyDays":30, "Frequency":4, "Monetary":120}, {
            "PredictedNextSegment":"At Risk", "PredictedNextSegmentProbability":0.8,
            "SHAPContributionsNextSegment":{"R_3m":0.4,"F_3m":-0.1}
        })
        joined = " ".join(result["evidence"])
        self.assertIn("At Risk", joined)
        self.assertIn("recency over 3 months", joined)
        self.assertEqual(result["priority"], "Review")

if __name__ == "__main__": unittest.main()
