import unittest
import numpy as np
from similarity_Measure_Color_Profile import compare_histograms


class TestColorHistogramSimilarity(unittest.TestCase):

    def setUp(self):
        # Create mock histograms for testing
        self.hist1 = np.random.rand(16, 16, 16).astype(np.float32)
        self.hist2 = np.random.rand(16, 16, 16).astype(np.float32)
        self.hist_identical = (
            self.hist1.copy()
        )  # Identical histogram for perfect similarity

    def test_compare_histograms_correlation(self):
        # Test using the correlation method
        result = compare_histograms(
            self.hist1, self.hist_identical, method="correlation"
        )
        self.assertAlmostEqual(abs(result), 1.0, places=5)  # Absolute value close to 1

    def test_compare_histograms_chi_square(self):
        # Test using the chi-square method
        result = compare_histograms(
            self.hist1, self.hist_identical, method="chi-square"
        )
        self.assertAlmostEqual(
            result, 0.0, places=5
        )  # Chi-square of identical histograms should be close to 0

    def test_compare_histograms_intersection(self):
        # Test using the intersection method
        result = compare_histograms(
            self.hist1, self.hist_identical, method="intersection"
        )
        self.assertAlmostEqual(
            result, np.sum(self.hist1), places=4
        )  # Relax precision to 4 places

    def test_compare_histograms_bhattacharyya(self):
        # Test using the Bhattacharyya method
        result = compare_histograms(
            self.hist1, self.hist_identical, method="bhattacharyya"
        )
        self.assertAlmostEqual(
            result, 0.0, places=5
        )  # Bhattacharyya distance between identical histograms should be close to 0

    def test_compare_histograms_different_shapes(self):
        # Test with histograms of different shapes
        hist_diff_shape = np.random.rand(8, 8, 8).astype(np.float32)
        with self.assertRaises(ValueError):
            compare_histograms(self.hist1, hist_diff_shape, method="correlation")


if __name__ == "__main__":
    unittest.main()
