import unittest
from unittest.mock import patch

import pandas as pd

from app.ml.temporal import training_samples


class TemporalLabelTests(unittest.TestCase):
    def setUp(self):
        self.states = pd.DataFrame([
            {"Customer ID": 1, "Month": pd.Period("2020-01", freq="M"), "Segment": "Loyal"},
            {"Customer ID": 1, "Month": pd.Period("2020-02", freq="M"), "Segment": "At Risk"},
            {"Customer ID": 1, "Month": pd.Period("2020-04", freq="M"), "Segment": "At Risk"},
            {"Customer ID": 2, "Month": pd.Period("2020-01", freq="M"), "Segment": "Champions"},
            {"Customer ID": 2, "Month": pd.Period("2020-02", freq="M"), "Segment": "Loyal"},
        ])
        self.transactions = pd.DataFrame([
            {"Customer ID": 1, "InvoiceDate": pd.Timestamp("2020-01-10")},
            {"Customer ID": 1, "InvoiceDate": pd.Timestamp("2020-04-30")},
            {"Customer ID": 2, "InvoiceDate": pd.Timestamp("2020-01-10")},
            # The Jan snapshot's 90-day horizon ends on this last observed date.
            {"Customer ID": 2, "InvoiceDate": pd.Timestamp("2020-05-01")},
        ])

    @patch("app.ml.temporal.purchase_rows")
    @patch("app.ml.temporal.temporal_snapshots")
    def test_next_segment_uses_only_the_immediately_following_month(self, snapshots, purchases):
        snapshots.return_value = self.states
        purchases.return_value = self.transactions
        result = training_samples()
        january = result[result["Month"] == pd.Period("2020-01", freq="M")].set_index("Customer ID")
        april = result[(result["Customer ID"] == 1) & (result["Month"] == pd.Period("2020-04", freq="M"))].iloc[0]
        self.assertEqual(january.loc[1, "NextSegment"], "At Risk")
        self.assertEqual(january.loc[2, "NextSegment"], "Loyal")
        self.assertTrue(pd.isna(april["NextSegment"]))

    @patch("app.ml.temporal.purchase_rows")
    @patch("app.ml.temporal.temporal_snapshots")
    def test_90_day_label_requires_full_horizon_and_uses_half_open_interval(self, snapshots, purchases):
        snapshots.return_value = self.states
        purchases.return_value = self.transactions
        result = training_samples()
        january = result[result["Month"] == pd.Period("2020-01", freq="M")].set_index("Customer ID")
        self.assertEqual(int(january.loc[1, "Churn90d"]), 0)  # Purchase occurred before day 90.
        self.assertEqual(int(january.loc[2, "Churn90d"]), 1)  # Purchase exactly at the exclusive horizon boundary.
        april = result[(result["Customer ID"] == 1) & (result["Month"] == pd.Period("2020-04", freq="M"))].iloc[0]
        self.assertTrue(pd.isna(april["Churn90d"]))  # Future 90-day horizon is not fully observed.


if __name__ == "__main__":
    unittest.main()
