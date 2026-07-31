import unittest

from llx2026.model import ShrinkageInputs
from llx2026.plotting import development_figure


class PlottingTests(unittest.TestCase):
    def test_development_figure_contains_mean_and_two_bands(self):
        figure = development_figure(
            ShrinkageInputs(100, 7, 60, 50, 0.45, 1860),
            maximum_time=365,
            points=40,
        )
        axis = figure.axes[0]
        self.assertGreaterEqual(len(axis.lines), 2)
        self.assertEqual(len(axis.collections), 3)

    def test_short_curve_is_rejected(self):
        with self.assertRaises(ValueError):
            development_figure(
                ShrinkageInputs(1, 7, 60, 50, 0.45, 1860),
                maximum_time=1,
            )


if __name__ == "__main__":
    unittest.main()
