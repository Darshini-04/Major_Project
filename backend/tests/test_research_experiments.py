import unittest

import pandas as pd

from app.ml.research_experiments import _train_rows


class PurgedTemporalSplitTests(unittest.TestCase):
    def test_next_segment_training_target_must_precede_test_month(self):
        months = pd.PeriodIndex(["2020-01", "2020-02", "2020-03"], freq="M")
        frame = pd.DataFrame({
            "Month": months,
            "NextMonth": months + 1,
            "NextSegment": ["Loyal", "At Risk", "Champions"],
        })
        train = _train_rows(frame, "NextSegment", pd.Period("2020-03", freq="M"))
        self.assertEqual(train["Month"].tolist(), [pd.Period("2020-01", freq="M")])

    def test_inactivity_training_horizon_must_end_by_test_cutoff(self):
        months = pd.PeriodIndex(["2020-01", "2020-02", "2020-03", "2020-04"], freq="M")
        frame = pd.DataFrame({"Month": months, "Churn90d": [0, 1, 0, 1]})
        train = _train_rows(frame, "Churn90d", pd.Period("2020-06", freq="M"))
        self.assertEqual(train["Month"].tolist(), [pd.Period("2020-01", freq="M"), pd.Period("2020-02", freq="M")])


if __name__ == "__main__":
    unittest.main()
