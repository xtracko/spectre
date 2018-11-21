import numpy as np
import scipy.signal as signal


def mean_peak_width(xic: np.ndarray) -> float:
    widths = []
    for x in np.transpose(xic):
        peaks, _ = signal.find_peaks(x)
        widths.extend(signal.peak_widths(x, peaks)[0])
    return float(np.mean(widths))


def preprocess(xic: np.ndarray, peak_width: float) -> np.ndarray:
    window = int(round(peak_width)) | 1  # make odd number
    degree = 3 if window > 3 else (window - 1)

    if degree > 0:
        xic = signal.savgol_filter(xic, window, degree, axis=0)
    xic[:, np.nanmax(xic, axis=0) < 100] = 0
    xic.clip(0, None, out=xic)

    return _remove_baseline(xic, int(10 * peak_width))


def _remove_baseline(xic: np.ndarray, k: int) -> np.ndarray:
    from spectre.dense.utils import rolling_min, rolling_mean, rolling_median

    k = k if k <= (len(xic) - 2) else (len(xic) - 2)
    k = k if (k % 2) else (k - 1)

    xic_pad = np.pad(xic, [[k // 2], [0]], mode='symmetric')
    xic_min = rolling_min(xic_pad, k)
    xic_min = xic_min + np.std(xic_min, axis=0)

    xic_base = rolling_median(xic_pad, k)
    xic_base[xic_base > xic_min] = xic_min[xic_base > xic_min]
    xic_base = np.pad(xic_base, [[k // 2], [0]], mode='symmetric')
    xic_base = rolling_mean(xic_base, k)
    return (xic - xic_base).clip(0, None)
