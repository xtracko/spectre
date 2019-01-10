from spectre import Xic


def mean_peak_width():
    raise NotImplementedError


def remove_noise(data, peak_width: float):
    peak_width = int(round(peak_width))
    window = (peak_width + 1) if (peak_width % 2) else peak_width
    degree = 3 if window > 3 else (window - 1)

    if degree > 0:
        data = savgol_filter(data, window, degree, axis=1)
    return data[data > 0]

def remove_baseline(xic, k: int):
    k = k if k <= (xic.shape[] - 2) else (xic.shape[] - 2)
    k = k if (k % 2) else (k - 1)

    raise NotImplementedError

def preprocess(xic: Xic, peak_width: float) -> Xic:
    xic.data = remove_noise(xic.data, peak_width)
    xic.data = remove_baseline(xic.data, int(10 * peak_width))
    return xic

################################################################################

def savgol_filter(x, window: int, degree: int, axis: int):
    raise NotImplementedError


def moving_min(x, window: int, axis: int):
    raise NotImplementedError


def moving_mean(x, window: int, axis: int):
    raise NotImplementedError


def moving_median(x, window: int, axis: int):
    raise NotImplementedError

################################################################################