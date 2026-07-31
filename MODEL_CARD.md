import unittest

import numpy as np

from llx2026.model import (
    PI_HALF_WIDTHS,
    aggregate_volume_fraction,
    predict,
    prediction_interval,
    predict_with_intervals,
)


class ModelTests(unittest.TestCase):
    def test_reference_gui_case(self):
        value = float(
            predict(
                drying_time=100,
                curing_age=7,
                relative_humidity=60,
                volume_to_surface=50,
                water_cement_ratio=0.45,
                aggregate_content=1860,
            )
        )
        self.assertAlmostEqual(value, 375.323, places=3)

    def test_aggregate_conversion_is_not_silently_clipped(self):
        self.assertAlmostEqual(float(aggregate_volume_fraction(1860)), 1860 / 2650)
        with self.assertRaises(ValueError):
            aggregate_volume_fraction(3000)

    def test_prediction_intervals_are_ordered(self):
        result = predict_with_intervals(
            drying_time=100,
            curing_age=7,
            relative_humidity=60,
            volume_to_surface=50,
            water_cement_ratio=0.45,
            aggregate_content=1860,
        )
        self.assertLess(result["pi95_lower"], result["pi90_lower"])
        self.assertLess(result["pi90_lower"], result["prediction"])
        self.assertLess(result["prediction"], result["pi90_upper"])
        self.assertLess(result["pi90_upper"], result["pi95_upper"])
        self.assertAlmostEqual(
            result["prediction"] - result["pi90_lower"],
            108.6639,
            places=4,
        )
        self.assertAlmostEqual(
            result["pi95_upper"] - result["prediction"],
            144.0709,
            places=4,
        )

    def test_prediction_interval_age_bins(self):
        mean = np.full(9, 500.0)
        ages = np.array([1.0, 7.0, 7.01, 28.0, 28.01, 180.0, 180.01, 1000.0, 1000.01])
        lower, upper = prediction_interval(mean, ages, 0.90)
        expected = np.array(
            [
                PI_HALF_WIDTHS[0, 1],
                PI_HALF_WIDTHS[0, 1],
                PI_HALF_WIDTHS[1, 1],
                PI_HALF_WIDTHS[1, 1],
                PI_HALF_WIDTHS[2, 1],
                PI_HALF_WIDTHS[2, 1],
                PI_HALF_WIDTHS[3, 1],
                PI_HALF_WIDTHS[3, 1],
                PI_HALF_WIDTHS[4, 1],
            ]
        )
        np.testing.assert_allclose(mean - lower, expected)
        np.testing.assert_allclose(upper - mean, expected)

    def test_unsupported_interval_level_is_rejected(self):
        with self.assertRaises(ValueError):
            prediction_interval(500.0, 100.0, 0.80)

    def test_vectorized_prediction(self):
        values = predict(
            drying_time=np.array([7.0, 28.0, 365.0]),
            curing_age=7,
            relative_humidity=50,
            volume_to_surface=22.7,
            water_cement_ratio=0.5,
            aggregate_content=1860,
        )
        self.assertEqual(values.shape, (3,))
        self.assertTrue(np.all(np.diff(values) > 0))


if __name__ == "__main__":
    unittest.main()
