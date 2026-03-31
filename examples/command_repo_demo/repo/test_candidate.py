from __future__ import annotations

import math
import unittest

from candidate import predict


class CandidateTests(unittest.TestCase):
    def test_predict_returns_finite_values(self) -> None:
        for x_value in (-2.0, -1.0, 0.0, 1.0, 2.0):
            prediction = predict(x_value)
            self.assertIsInstance(prediction, float)
            self.assertTrue(math.isfinite(prediction))

    def test_predict_stays_within_sane_bounds(self) -> None:
        for x_value in (-10.0, -2.0, 0.0, 2.0, 10.0):
            self.assertLess(abs(predict(x_value)), 50.0)


if __name__ == "__main__":
    unittest.main()
