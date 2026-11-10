import unittest

import numpy as np

from app.ml.train import _binary_calibration_metrics, _multiclass_calibration_metrics


class CalibrationMetricTests(unittest.TestCase):
    def test_binary_brier_and_expected_calibration_error(self):
        result = _binary_calibration_metrics([0, 1, 0, 1], [0.1, 0.8, 0.4, 0.7])
        self.assertAlmostEqual(result["brier_score"], 0.075)
        self.assertAlmostEqual(result["expected_calibration_error"], 0.25)
        self.assertEqual(sum(item["count"] for item in result["calibration_bins"]), 4)

    def test_multiclass_brier_and_top_label_calibration(self):
        result = _multiclass_calibration_metrics([0, 1], [[0.8, 0.2], [0.3, 0.7]])
        self.assertAlmostEqual(result["multiclass_brier_score"], 0.13)
        self.assertAlmostEqual(result["top_label_ece"], 0.25)
        self.assertAlmostEqual(result["mean_top_class_confidence"], 0.75)

    def test_probability_one_is_included_in_last_bin(self):
        result = _binary_calibration_metrics([1], [1.0])
        self.assertEqual(result["calibration_bins"][-1]["count"], 1)


if __name__ == "__main__":
    unittest.main()
