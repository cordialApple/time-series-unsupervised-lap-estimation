import numpy as np
import pandas as pd


def _replace_nan_with_mean(values):
    return np.nan_to_num(values, nan=np.nanmean(values))


def robust_standardize(values):
    array = np.asarray(values, dtype=float)
    one_dimensional = array.ndim == 1
    if one_dimensional:
        array = array[:, None]
    median = np.nanmedian(array, axis=0)
    q25 = np.nanquantile(array, 0.25, axis=0)
    q75 = np.nanquantile(array, 0.75, axis=0)
    scale = q75 - q25
    fallback = np.nanstd(array, axis=0)
    scale = np.where(scale > 1e-12, scale, fallback)
    scale = np.where(scale > 1e-12, scale, 1.0)
    result = (array - median) / scale
    result = np.nan_to_num(result)
    return result[:, 0] if one_dimensional else result


def moving_average(values, window_size):
    array = np.asarray(values, dtype=float)
    if window_size <= 1:
        return array.copy()
    kernel = np.ones(int(window_size), dtype=float) / int(window_size)
    if array.ndim == 1:
        return np.convolve(array, kernel, mode="same")
    return np.column_stack([np.convolve(array[:, index], kernel, mode="same") for index in range(array.shape[1])])


def build_multivariate_trace(frame, channels, smooth_samples=5):
    if not channels:
        raise ValueError("At least one informative channel is required")
    numeric = frame[channels].apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both")
    standardized = robust_standardize(numeric.to_numpy(dtype=float))
    smoothed = moving_average(standardized, smooth_samples)
    return np.mean(smoothed, axis=1)


def normalized_autocorrelation(values, max_lag_samples=None):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Autocorrelation input must be one-dimensional")
    array = _replace_nan_with_mean(array)
    lag_count = len(array) if max_lag_samples is None else min(len(array), int(max_lag_samples) + 1)
    correlation = np.zeros(lag_count, dtype=float)
    for lag in range(lag_count):
        first = array[: len(array) - lag] if lag else array
        second = array[lag:]
        first = first - first.mean()
        second = second - second.mean()
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        correlation[lag] = np.dot(first, second) / denominator if denominator > 1e-12 else 0.0
    return correlation


def rank_autocorrelation_peaks(correlation, sample_rate_hz, min_lag_s, max_lag_s, top_k=5):
    values = np.asarray(correlation, dtype=float)
    lower = max(1, int(np.ceil(min_lag_s * sample_rate_hz)))
    upper = min(len(values) - 2, int(np.floor(max_lag_s * sample_rate_hz)))
    peaks = []
    for index in range(lower, upper + 1):
        if values[index] > values[index - 1] and values[index] >= values[index + 1]:
            peaks.append((index / sample_rate_hz, values[index]))
    peaks.sort(key=lambda item: item[1], reverse=True)
    return pd.DataFrame(peaks[:top_k], columns=["lag_s", "correlation"])


def dominant_fft_periods(values, sample_rate_hz, min_period_s, max_period_s, top_k=5):
    array = np.asarray(values, dtype=float)
    centered = np.nan_to_num(array - np.nanmean(array))
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)
    power = np.abs(np.fft.rfft(centered)) ** 2
    positive = frequencies > 0
    periods = np.divide(1.0, frequencies, out=np.full_like(frequencies, np.inf), where=positive)
    allowed = positive & (periods >= min_period_s) & (periods <= max_period_s)
    indexes = np.flatnonzero(allowed)
    indexes = indexes[np.argsort(power[indexes])[::-1]][:top_k]
    return pd.DataFrame({"period_s": periods[indexes], "power": power[indexes]})


def lagged_cross_correlation(first, second, max_lag_samples):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.ndim != 1 or second.ndim != 1 or len(first) != len(second):
        raise ValueError("Cross-correlation inputs must be equal-length one-dimensional arrays")
    rows = []
    for lag in range(-int(max_lag_samples), int(max_lag_samples) + 1):
        if lag < 0:
            aligned_first = first[-lag:]
            aligned_second = second[: len(second) + lag]
        elif lag > 0:
            aligned_first = first[: len(first) - lag]
            aligned_second = second[lag:]
        else:
            aligned_first = first
            aligned_second = second
        aligned_first = _replace_nan_with_mean(aligned_first)
        aligned_second = _replace_nan_with_mean(aligned_second)
        aligned_first = aligned_first - aligned_first.mean()
        aligned_second = aligned_second - aligned_second.mean()
        denominator = np.linalg.norm(aligned_first) * np.linalg.norm(aligned_second)
        correlation = np.dot(aligned_first, aligned_second) / denominator if denominator > 1e-12 else 0.0
        rows.append({"lag_samples": lag, "correlation": correlation, "overlap_samples": len(aligned_first)})
    return pd.DataFrame(rows)


def recurrence_candidates(frame, channels, sample_rate_hz, min_lag_s=15.0, max_lag_s=120.0, top_k=5):
    trace = build_multivariate_trace(frame, channels)
    duration_s = (len(trace) - 1) / sample_rate_hz
    usable_max = min(max_lag_s, duration_s / 2.0)
    if usable_max <= min_lag_s:
        return pd.DataFrame(columns=["lag_s", "correlation", "overlap_s", "cycle_count"])
    correlation = normalized_autocorrelation(trace, max_lag_samples=round(usable_max * sample_rate_hz))
    peaks = rank_autocorrelation_peaks(correlation, sample_rate_hz, min_lag_s, usable_max, top_k)
    if peaks.empty:
        return peaks.assign(overlap_s=pd.Series(dtype=float), cycle_count=pd.Series(dtype=float))
    peaks["overlap_s"] = duration_s - peaks["lag_s"]
    peaks["cycle_count"] = duration_s / peaks["lag_s"]
    return peaks
