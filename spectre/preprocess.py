from typing import Callable, Union, Optional

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

from .xic import Xic


SparseMatrix = Union[csr_matrix, csc_matrix]


def std(x: SparseMatrix, axis: int) -> np.ndarray:
    if axis not in (0, 1):
        raise ValueError(f"Unsupported axis value {axis} for 2 dimensional matrix")

    x = x.tocsc() if axis == 0 else x.tocsr()
    y = np.zeros(shape=x.shape[1 - axis], dtype=x.dtype)

    for i in range(x.shape[1 - axis]):
        xa = x.getcol(i) if axis == 0 else x.getrow(i)
        y[i] = np.std(np.squeeze(xa.toarray()))
    return y


def _apply(x: SparseMatrix, axis: int, func: Callable) -> SparseMatrix:
    if axis not in (0, 1):
        raise ValueError(f"Unsupported axis value {axis} for 2 dimensional matrix")

    xn = 0
    m, n = x.shape
    data, indices, pointers = [], [], [0]
    x = x.tocsc() if axis == 0 else x.tocsr()

    for i in range(n if axis == 0 else m):
        xa = x.getcol(i) if axis == 0 else x.getrow(i)
        xa = func(np.squeeze(xa.toarray()))
        xi = np.flatnonzero(xa)
        xn = max(xn, xa.shape[0])

        data.extend(xa[xi])
        indices.extend(xi)
        pointers.append(len(indices))

    shape = (xn, n) if axis == 0 else (m, xn)
    cls = csc_matrix if axis == 0 else csr_matrix
    return cls((data, indices, pointers), shape=shape)


def rolling_min(x: SparseMatrix, k: int, axis: int = 0, mode: Optional[str] = None) -> SparseMatrix:
    from .dense.utils import rolling_min
    return _apply(x=x, axis=axis, func=lambda a: rolling_min(a, k=k, axis=0, mode=mode))


def rolling_max(x: SparseMatrix, k: int, axis: int = 0, mode: Optional[str] = None) -> SparseMatrix:
    from .dense.utils import rolling_max
    return _apply(x=x, axis=axis, func=lambda a: rolling_max(a, k=k, axis=0, mode=mode))


def rolling_mean(x: SparseMatrix, k: int, axis: int = 0, mode: Optional[str] = None) -> SparseMatrix:
    from .dense.utils import rolling_mean
    return _apply(x=x, axis=axis, func=lambda a: rolling_mean(a, k=k, axis=0, mode=mode))


def rolling_median(x: SparseMatrix, k: int, axis: int = 0, mode: Optional[str] = None) -> SparseMatrix:
    from .dense.utils import rolling_median
    return _apply(x=x, axis=axis, func=lambda a: rolling_median(a, k=k, axis=0, mode=mode))


def savgol_filter(x: SparseMatrix, window: int, degree: int, axis: int) -> SparseMatrix:
    from scipy.signal import savgol_filter
    return _apply(x=x, axis=axis, func=lambda a: savgol_filter(a, window_length=window, polyorder=degree))


def remove_noise(data: SparseMatrix, peak_width: float) -> SparseMatrix:
    peak_width = int(round(peak_width))
    window = peak_width if (peak_width % 2) else (peak_width + 1)
    degree = 3 if window > 3 else (window - 1)

    if degree > 0:
        data = savgol_filter(data, window, degree, axis=0)

    # TODO: implement clipping of items <0 and then pruning of an array in 1-pass algorithm (currently at least 3 pass)
    data[data < 0] = 0
    data.eliminate_zeros()
    data.sum_duplicates()
    return data


def remove_baseline(data: SparseMatrix, k: int) -> SparseMatrix:
    k = k if k <= (data.shape[0] - 2) else (data.shape[0] - 2)
    k = k if (k % 2) else (k - 1)

    data_min = rolling_min(data, k, axis=0, mode='symmetric')
    data_min = data_min + std(data_min, axis=0)

    data_base = rolling_median(data, k, axis=0, mode='symmetric')
    data_base[data_base > data_min] = data_min[data_base > data_min]
    data_base = rolling_mean(data_base, k, axis=0, mode='symmetric')

    data = data - data_base
    data[data < 0] = 0
    data.eliminate_zeros()
    data.sum_duplicates()
    return data


def mean_peak_width(xic: Xic) -> float:
    from scipy.signal import find_peaks, peak_widths

    widths = []
    data = xic.data.tocsc()
    for i in range(data.shape[1]):
        x = np.squeeze(data.getcol(i).toarray())
        peaks, _ = find_peaks(x)
        widths.extend(peak_widths(x, peaks)[0])
    return float(np.mean(widths))


def preprocess(xic: Xic, peak_width: float) -> Xic:
    xic.data = remove_noise(xic.data, peak_width)
    xic.data = remove_baseline(xic.data, int(10 * peak_width))
    return xic
