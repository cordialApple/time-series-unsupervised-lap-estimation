import unittest

import numpy as np
import pandas as pd


class SignalTests(unittest.TestCase):
    def test_autocorrelation_recovers_known_period(self):
        from lap_estimation.signal import normalized_autocorrelation, rank_autocorrelation_peaks

        sample_rate_hz = 10.0
        period_s = 12.0
        time_s = np.arange(0.0, 72.0, 1.0 / sample_rate_hz)
        signal = np.sin(2.0 * np.pi * time_s / period_s)

        correlation = normalized_autocorrelation(signal)
        peaks = rank_autocorrelation_peaks(
            correlation,
            sample_rate_hz=sample_rate_hz,
            min_lag_s=8.0,
            max_lag_s=20.0,
            top_k=3,
        )

        self.assertAlmostEqual(peaks.iloc[0]["lag_s"], period_s, delta=0.2)
        self.assertGreater(peaks.iloc[0]["correlation"], 0.7)

    def test_robust_standardize_handles_constant_channel(self):
        from lap_estimation.signal import robust_standardize

        values = np.column_stack([np.arange(5.0), np.ones(5)])

        standardized = robust_standardize(values)

        self.assertTrue(np.isfinite(standardized).all())
        self.assertTrue(np.allclose(standardized[:, 1], 0.0))

    def test_cross_correlation_uses_positive_lag_for_following_signal(self):
        from lap_estimation.signal import lagged_cross_correlation

        source = np.zeros(40)
        source[10:15] = 1.0
        follower = np.zeros(40)
        follower[13:18] = 1.0

        result = lagged_cross_correlation(source, follower, max_lag_samples=6)
        peak = result.iloc[result["correlation"].argmax()]

        self.assertEqual(peak["lag_samples"], 3)

    def test_recurrence_candidates_require_two_full_cycles(self):
        from lap_estimation.signal import recurrence_candidates

        frame = pd.DataFrame({"signal": np.random.default_rng(42).normal(size=400)})

        candidates = recurrence_candidates(frame, ["signal"], sample_rate_hz=10.0, min_lag_s=15.0, max_lag_s=120.0, top_k=100)

        self.assertLessEqual(candidates["lag_s"].max(), 20.0)
        self.assertGreaterEqual(candidates["cycle_count"].min(), 2.0)


if __name__ == "__main__":
    unittest.main()
