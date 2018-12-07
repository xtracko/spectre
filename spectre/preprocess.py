from spectre import Xic


def find_peaks():
    raise NotImplementedError


def peak_bases():
    raise NotImplementedError


def mean_peak_width():
    raise NotImplementedError


def savgol_filter(x, window: int, degree: int, axis: int):
    raise NotImplementedError


def moving_min():
    raise NotImplementedError


def moving_mean():
    raise NotImplementedError


def moving_median():
    raise NotImplementedError


def _make_odd(number: int) -> int:
    return (number + 1) if number % 2 else number


def preprocess(xic: Xic, peak_width: float) -> Xic:
    window = _make_odd(int(round(peak_width)))
    degree = 3 if window > 3 else (window - 1)

    if degree > 0:
        xic.data = sparse.savgol_filter(xic.data, window, degree, axis=1)
    sparse.remove_if_nonpositive(xic.data)

    sparse.remove_baseline()
    return xic
