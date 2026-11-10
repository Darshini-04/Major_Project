import unittest
import pandas as pd
from app.ml.segmentation import segment_customers

class SegmentationTests(unittest.TestCase):
    def test_scores_reward_recent_frequent_high_value_customers(self):
        df = pd.DataFrame({"RecencyDays": [1, 10, 20, 30, 40], "Frequency": [5, 4, 3, 2, 1], "Monetary": [500, 400, 300, 200, 100]})
        out = segment_customers(df)
        self.assertEqual(out.iloc[0].R_Score, 5)
        self.assertEqual(out.iloc[0].F_Score, 5)
        self.assertEqual(out.iloc[0].M_Score, 5)
        self.assertEqual(out.iloc[-1].R_Score, 1)

    def test_empty_input_is_supported(self):
        out = segment_customers(pd.DataFrame(columns=["RecencyDays", "Frequency", "Monetary"]))
        self.assertTrue(out.empty)
        self.assertIn("Segment", out.columns)

if __name__ == "__main__": unittest.main()
