import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import uniform_filter1d

from .signal import normalized_autocorrelation


FUNDAMENTAL_FRACTION = 0.8


def block_shuffle(values, block_samples, rng):
    values = np.asarray(values)
    blocks = np.array_split(values, max(1, len(values) // int(block_samples)))
    return np.concatenate([blocks[index] for index in rng.permutation(len(blocks))])


def strongest_recurrence(trace, min_lag_samples, max_lag_samples):
    correlation = normalized_autocorrelation(trace, max_lag_samples)
    lags = np.arange(max(1, int(min_lag_samples)), min(len(correlation) - 2, int(max_lag_samples)) + 1)
    lags = lags[(correlation[lags] > correlation[lags - 1]) & (correlation[lags] >= correlation[lags + 1])]
    if not len(lags):
        return None
    strongest = float(correlation[lags].max())
    # Multiples of a true period peak equally high; report shortest comparable lag.
    comparable = lags[correlation[lags] >= FUNDAMENTAL_FRACTION * strongest] if strongest > 0 else lags[[correlation[lags].argmax()]]
    return int(comparable.min()), strongest


def calibrated_recurrence(trace, min_lag_samples, max_lag_samples, block_samples, surrogate_count, rng):
    trace = np.nan_to_num(np.asarray(trace, dtype=float))
    max_lag_samples = min(int(max_lag_samples), len(trace) // 2)
    if max_lag_samples <= min_lag_samples or np.ptp(trace) <= 1e-12:
        return None
    # Pace and level changes slower than shortest lap pass as recurrence against a block-shuffle null; keep within-lap detail only.
    trace = trace - uniform_filter1d(trace, int(min_lag_samples), mode="nearest")
    observed = strongest_recurrence(trace, min_lag_samples, max_lag_samples)
    if observed is None:
        return None
    lag_samples, correlation = observed
    null = np.full(surrogate_count, -1.0)
    for index in range(surrogate_count):
        surrogate = strongest_recurrence(block_shuffle(trace, block_samples, rng), min_lag_samples, max_lag_samples)
        if surrogate is not None:
            null[index] = surrogate[1]
    return {
        "lag_samples": lag_samples,
        "correlation": correlation,
        "p_value": float((1 + np.sum(null >= correlation)) / (1 + surrogate_count)),
        "null_q95": float(np.quantile(null, 0.95)),
        "at_search_ceiling": bool(lag_samples >= 0.9 * max_lag_samples),
    }


def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(p_values)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def resample_by_distance(distance, values, sample_count):
    distance = np.asarray(distance, dtype=float)
    values = np.asarray(values, dtype=float)
    advancing = np.concatenate(([True], np.diff(np.maximum.accumulate(distance)) > 0))
    distance = distance[advancing]
    values = values[advancing]
    if len(distance) < 2:
        raise ValueError("Distance resampling requires forward motion")
    grid = np.linspace(distance[0], distance[-1], int(sample_count))
    return grid, np.interp(grid, distance, values)


def heading_closure(yaw_rate_deg_s, sample_rate_hz, lag_samples, min_lag_samples, max_lag_samples):
    heading = cumulative_trapezoid(np.nan_to_num(yaw_rate_deg_s), dx=1.0 / sample_rate_hz, initial=0.0)

    def turn_statistics(lag):
        turn = heading[lag:] - heading[:-lag]
        median = np.median(turn)
        return median, np.median(np.abs(turn - median))

    upper = min(int(max_lag_samples), len(heading) // 2)
    spreads = np.array([turn_statistics(lag)[1] for lag in range(int(min_lag_samples), upper + 1)])
    median, spread = turn_statistics(int(lag_samples))
    return {
        "turn_per_period_deg": float(median),
        "turn_spread_deg": float(spread),
        "turn_spread_percentile": float(np.mean(spreads <= spread)),
        "heading_range_turns": float(np.ptp(heading) / 360.0),
    }


def tile_warped_laps(template, lap_count, warp_fraction, noise_fraction, rng):
    template = np.asarray(template, dtype=float)
    source = np.arange(len(template))
    laps = []
    for _ in range(lap_count):
        length = max(2, int(round(len(template) * rng.uniform(1.0 - warp_fraction, 1.0 + warp_fraction))))
        laps.append(np.interp(np.linspace(0, len(template) - 1, length), source, template))
    signal = np.concatenate(laps)
    noise = rng.normal(0.0, noise_fraction * np.std(template), len(signal))
    return signal + noise, len(signal) / lap_count


def segment_runs(yaw_rate_deg_s, sample_rate_hz, activity_threshold=5.0, window_s=3.0, max_gap_s=5.0, min_run_s=30.0):
    yaw_rate = np.nan_to_num(np.asarray(yaw_rate_deg_s, dtype=float))
    activity = np.sqrt(np.maximum(uniform_filter1d(yaw_rate**2, int(window_s * sample_rate_hz), mode="nearest"), 0.0))
    active = np.concatenate(([0], (activity > activity_threshold).astype(int), [0]))
    runs = []
    for start, stop in np.flatnonzero(np.diff(active)).reshape(-1, 2):
        if runs and start - runs[-1][1] <= max_gap_s * sample_rate_hz:
            runs[-1][1] = stop
        else:
            runs.append([start, stop])
    return [(int(start), int(stop)) for start, stop in runs if stop - start >= min_run_s * sample_rate_hz]


def align_run(reference, run, scales, max_shift_samples, shift_step, min_overlap=0.6):
    reference = np.asarray(reference, dtype=float)
    run = np.asarray(run, dtype=float)
    source = np.arange(len(run))
    best = (-1.0, 1.0, 0)
    for scale in scales:
        length = int(round(len(run) * scale))
        stretched = np.interp(np.linspace(0, len(run) - 1, length), source, run)
        for shift in range(-int(max_shift_samples), int(max_shift_samples) + 1, int(shift_step)):
            start = max(0, shift)
            stop = min(len(reference), shift + length)
            if stop - start < min_overlap * min(len(reference), length):
                continue
            correlation = np.corrcoef(reference[start:stop], stretched[start - shift : stop - shift])[0, 1]
            if correlation > best[0]:
                best = (float(correlation), float(scale), shift)
    return best
