import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi import HTTPException

from app.main import customers, download_model_evaluation, download_predictions


class PredictionRouteTests(unittest.TestCase):
    def test_customer_filter_limits_results_to_prediction_rows(self):
        profiles = pd.DataFrame({
            "Customer ID": [10, 20, 30], "Country": ["UK"] * 3, "RecencyDays": [5, 10, 15],
            "Frequency": [1, 2, 3], "Monetary": [10.0, 20.0, 30.0], "R_Score": [1, 2, 3],
            "F_Score": [1, 2, 3], "M_Score": [1, 2, 3], "Segment": ["Loyal"] * 3,
            "InterPurchaseDays": [None] * 3, "TenureDays": [0] * 3, "AvgBasketValue": [10.0, 10.0, 10.0],
        })
        with tempfile.TemporaryDirectory() as directory:
            pd.DataFrame({"Customer ID": [20, 30]}).to_csv(Path(directory) / "current_predictions.csv", index=False)
            with patch("app.main.rfm", return_value=profiles), patch("app.main.OUTPUT_DIR", Path(directory)):
                result = customers(activity_period_days=None, has_prediction=True, limit=100)
        self.assertEqual(result["total"], 2)
        self.assertEqual([row["Customer ID"] for row in result["items"]], [30, 20])

    def test_prediction_and_evaluation_exports_return_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "current_predictions.csv").write_text("Customer ID\n10\n")
            Path(directory, "model_metrics.json").write_text("{}")
            with patch("app.main.OUTPUT_DIR", Path(directory)):
                predictions = download_predictions()
                evaluation = download_model_evaluation()
        self.assertEqual(predictions.filename, "pulseiq_customer_predictions.csv")
        self.assertEqual(evaluation.filename, "pulseiq_model_evaluation.json")

    def test_download_routes_report_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory, patch("app.main.OUTPUT_DIR", Path(directory)):
            with self.assertRaises(HTTPException) as prediction_error:
                download_predictions()
            with self.assertRaises(HTTPException) as evaluation_error:
                download_model_evaluation()
        self.assertEqual(prediction_error.exception.status_code, 404)
        self.assertEqual(evaluation_error.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
