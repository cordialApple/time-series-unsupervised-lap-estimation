import unittest

import numpy as np


def _colored_noise(rng, length, smooth=10):
    return np.convolve(rng.normal(size=length + smooth), np.ones(smooth) / smooth, mode="valid")[:length]


class PhysicsTests(unittest.TestCase):
    def test_block_shuffle_preserves_values(self):
        from lap_estimation.physics import block_shuffle

        values = np.arange(103.0)

        shuffled = block_shuffle(values, 10, np.random.default_rng(1))

        self.assertFalse(np.array_equal(shuffled, values))
        self.assertTrue(np.array_equal(np.sort(shuffled), values))

    def test_calibrated_recurrence_detects_repeated_laps(self):
        from lap_estimation.physics import calibrated_recurrence

        rng = np.random.default_rng(2)
        lap = _colored_noise(rng, 300)
        trace = np.tile(lap, 4) + 0.2 * rng.normal(size=1200)

        result = calibrated_recurrence(trace, 150, 1200, 50, 99, rng)

        self.assertAlmostEqual(result["lag_samples"], 300, delta=3)
        self.assertLessEqual(result["p_value"], 0.02)
        self.assertGreater(result["correlation"], result["null_q95"])

    def test_calibrated_recurrence_does_not_flag_unrelated_driving(self):
        from lap_estimation.physics import calibrated_recurrence

        rng = np.random.default_rng(3)
        p_values = []
        for _ in range(20):
            unrelated = np.concatenate([_colored_noise(rng, 300) + level for level in rng.normal(0.0, 2.0, 4)])
            p_values.append(calibrated_recurrence(unrelated, 150, 1200, 50, 99, rng)["p_value"])

        self.assertLessEqual(np.mean(np.array(p_values) < 0.05), 0.15)

    def test_strongest_recurrence_prefers_fundamental_over_multiple(self):
        from lap_estimation.physics import strongest_recurrence

        rng = np.random.default_rng(7)
        trace = np.tile(_colored_noise(rng, 200), 6) + 0.05 * rng.normal(size=1200)

        lag, _ = strongest_recurrence(trace, 100, 600)

        self.assertAlmostEqual(lag, 200, delta=2)

    def test_calibrated_recurrence_rejects_constant_and_short_traces(self):
        from lap_estimation.physics import calibrated_recurrence

        rng = np.random.default_rng(4)

        self.assertIsNone(calibrated_recurrence(np.zeros(1200), 150, 1200, 50, 9, rng))
        self.assertIsNone(calibrated_recurrence(rng.normal(size=200), 150, 1200, 50, 9, rng))

    def test_distance_resampling_removes_pace_variation(self):
        from lap_estimation.physics import calibrated_recurrence, resample_by_distance

        rng = np.random.default_rng(5)
        lap = _colored_noise(rng, 300) + 3.0
        fast = np.interp(np.linspace(0, 299, 250), np.arange(300), lap) * 300.0 / 250.0
        speed = np.concatenate([lap, fast, lap, fast])
        distance = np.cumsum(speed)

        grid, profile = resample_by_distance(distance, speed, len(speed))
        result = calibrated_recurrence(profile, 150, len(profile), 50, 19, rng)

        lap_distance = distance[-1] / 4.0
        self.assertAlmostEqual(result["lag_samples"] * (grid[1] - grid[0]), lap_distance, delta=0.03 * lap_distance)
        self.assertGreater(result["correlation"], 0.8)

    def test_distance_resampling_drops_standstill(self):
        from lap_estimation.physics import resample_by_distance

        grid, profile = resample_by_distance([0.0, 1.0, 1.0, 1.0, 2.0], [5.0, 6.0, 0.0, 0.0, 7.0], 3)

        self.assertTrue(np.allclose(grid, [0.0, 1.0, 2.0]))
        self.assertTrue(np.allclose(profile, [5.0, 6.0, 7.0]))

    def test_heading_closure_reports_full_turn_for_closed_circuit(self):
        from lap_estimation.physics import heading_closure

        sample_rate_hz = 20.0
        period_samples = 600
        phase = 2.0 * np.pi * np.arange(2400) / period_samples
        yaw_rate = 12.0 + 20.0 * np.sin(phase) + 8.0 * np.sin(3.0 * phase)

        closure = heading_closure(yaw_rate, sample_rate_hz, period_samples, 300, 1200)

        self.assertAlmostEqual(closure["turn_per_period_deg"], 360.0, delta=1.0)
        self.assertLess(closure["turn_spread_deg"], 1.0)
        self.assertLess(closure["turn_spread_percentile"], 0.05)
        self.assertAlmostEqual(closure["heading_range_turns"], 4.0, delta=0.2)

    def test_benjamini_hochberg_matches_known_adjustment(self):
        from lap_estimation.physics import benjamini_hochberg

        adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.5])

        self.assertTrue(np.allclose(adjusted, [0.04, 0.16 / 3.0, 0.16 / 3.0, 0.5]))

    def test_tiled_laps_report_mean_period(self):
        from lap_estimation.physics import tile_warped_laps

        signal, period = tile_warped_laps(np.sin(np.arange(100.0)), 4, 0.1, 0.0, np.random.default_rng(6))

        self.assertEqual(len(signal), round(period * 4))
        self.assertAlmostEqual(period, 100.0, delta=10.0)

    def test_segment_runs_merges_short_gaps_and_drops_idle(self):
        from lap_estimation.physics import segment_runs

        rng = np.random.default_rng(8)
        yaw_rate = np.zeros(4000)
        yaw_rate[500:1500] = 30.0 * rng.normal(size=1000)
        yaw_rate[1540:2400] = 30.0 * rng.normal(size=860)
        yaw_rate[3000:3200] = 30.0 * rng.normal(size=200)

        runs = segment_runs(yaw_rate, 20.0)

        self.assertEqual(len(runs), 1)
        self.assertAlmostEqual(runs[0][0], 500, delta=40)
        self.assertAlmostEqual(runs[0][1], 2400, delta=40)

    def test_align_run_recovers_pace_and_offset(self):
        from lap_estimation.physics import align_run

        rng = np.random.default_rng(9)
        course = _colored_noise(rng, 1600, smooth=40)
        slower = np.interp(np.linspace(0, 1599, 1760), np.arange(1600), course)[60:]
        scales = np.arange(0.8, 1.2001, 0.01)

        correlation, scale, shift = align_run(course, slower, scales, 200, 4)
        unrelated = align_run(course, _colored_noise(rng, 1700, smooth=40), scales, 200, 4)[0]

        self.assertGreater(correlation, 0.95)
        self.assertAlmostEqual(scale, 1600 / 1760, delta=0.02)
        self.assertAlmostEqual(shift, 60 * 1600 / 1760, delta=12)
        self.assertLess(unrelated, 0.5)

    def test_physics_figures_render_from_artifacts(self):
        import tempfile
        from pathlib import Path

        import matplotlib

        matplotlib.use("Agg")
        from lap_estimation.plotting_physics import render_null_calibration

        root = Path(__file__).resolve().parents[1] / "artifacts"
        with tempfile.TemporaryDirectory() as directory:
            path = render_null_calibration(root, Path(directory) / "calibration.png")

            self.assertGreater(path.stat().st_size, 20000)


if __name__ == "__main__":
    unittest.main()
