import tempfile
import unittest
from pathlib import Path

import pandas as pd

from llx2026.batch import BatchFormatError, OUTPUT_COLUMNS, predict_csv, predict_dataframe


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "dt": [7.0, 100.0],
                "t0": [7.0, 7.0],
                "RH": [50.0, 60.0],
                "VtoS": [22.7, 50.0],
                "wc": [0.50, 0.45],
                "agg_total": [1860.0, 1860.0],
                "record_id": ["A", "B"],
            }
        )

    def test_predictions_are_appended_without_changing_inputs(self):
        result = predict_dataframe(self.frame)
        self.assertEqual(result["record_id"].tolist(), ["A", "B"])
        for column in OUTPUT_COLUMNS:
            self.assertIn(column, result)
        self.assertAlmostEqual(result.loc[1, OUTPUT_COLUMNS[0]], 375.323, places=3)

    def test_missing_column_is_reported(self):
        with self.assertRaisesRegex(BatchFormatError, "aggregate_content"):
            predict_dataframe(self.frame.drop(columns="agg_total"))

    def test_csv_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "input.csv")
            target = Path(directory, "output.csv")
            self.frame.to_csv(source, index=False)
            result = predict_csv(source, target)
            self.assertTrue(target.is_file())
            self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
