from typing import Union

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse import isspmatrix_coo, isspmatrix_csr, isspmatrix_csc

from spectre.xic import Xic
from spectre.sparse import _sparse


def is_canonical(x: Union[coo_matrix, csr_matrix, csc_matrix]):
    if isspmatrix_coo(x):
        return _sparse.is_canonical_coo(x.row, x.col)
    elif isspmatrix_csr(x) or isspmatrix_csc(x):
        return _sparse.is_canonical_csr(x.indptr, x.indices)
    else:
        raise TypeError(f"unsupported type given '{type(x)}'")


def std(x: Union[csr_matrix, csc_matrix], axis: int):
    if axis not in (0, 1):
        raise ValueError(
            f"Unsupported axis value {axis} for 2 dimensional matrix")

    x = x.tocsc() if axis == 0 else x.tocsr()
    y = np.zeros(shape=x.shape[1 - axis], dtype=x.dtype)

    _sparse.std_csr(x.indptr, x.data, x.shape[axis], y)
    return y


def rolling(func, x: Union[csr_matrix, csc_matrix], window: int, axis: int):
    x = x.tocsr() if axis else x.tocsc()
    x.sum_duplicates()
    minor_size = x.shape[1] if axis else x.shape[0]

    shape = x.shape
    pointers = np.empty_like(x.indptr)
    size = _sparse.rolling_alloc_csr(x.indptr, x.indices, pointers, minor_size,
                                     window)
    data = np.zeros(size, dtype=x.data.dtype)
    indices = np.zeros(size, dtype=x.indices.dtype)

    func(x.indptr, x.indices, x.data, indices, data, minor_size, window)

    if axis:
        return csr_matrix((data, indices, pointers), shape)
    else:
        return csc_matrix((data, indices, pointers), shape)


def rolling_min(x: Union[csr_matrix, csc_matrix], window: int, axis: int):
    return rolling(_sparse.rolling_min_csr, x, window, axis)


def rolling_max(x: Union[csr_matrix, csc_matrix], window: int, axis: int):
    return rolling(_sparse.rolling_max_csr, x, window, axis)


def rolling_mean(x: Union[csr_matrix, csc_matrix], window: int, axis: int):
    return rolling(_sparse.rolling_mean_csr, x, window, axis)


def rolling_median(x: Union[csr_matrix, csc_matrix], window: int, axis: int):
    return rolling(_sparse.rolling_median_csr, x, window, axis)


def savgol_filter(x: Union[csr_matrix, csc_matrix], window: int, degree: int,
                  axis: int):
    x = x.tocsr() if axis else x.tocsc()
    x.sum_duplicates()

    minor_size = x.shape[1] if axis else x.shape[0]

    shape = x.shape
    pointers = np.empty_like(x.indptr)
    size = _sparse.rolling_alloc_csr(x.indptr, x.indices, pointers, minor_size,
                                     window)
    data = np.zeros(size, dtype=x.data.dtype)
    indices = np.zeros(size, dtype=x.indices.dtype)

    from scipy.signal import savgol_coeffs
    coeffs = savgol_coeffs(window, degree)
    _sparse.convolve_csr_dv(x.indptr, x.indices, x.data, coeffs, indices, data,
                            minor_size)

    if axis:
        return csr_matrix((data, indices, pointers), shape)
    else:
        return csc_matrix((data, indices, pointers), shape)


def remove_noise(data: Union[csr_matrix, csc_matrix], peak_width: float):
    peak_width = int(round(peak_width))
    window = peak_width if (peak_width % 2) else (peak_width + 1)
    degree = 3 if window > 3 else (window - 1)

    if degree > 0:
        data = savgol_filter(data, window, degree, axis=0)

    # IDEA: implement clipping of items <0 and then pruning of an array in
    # 1-pass algorithm (currently at least 3 pass)
    data[data < 0] = 0
    data.eliminate_zeros()
    data.sum_duplicates()
    return data


def remove_baseline(data: Union[csr_matrix, csc_matrix], k: int):
    k = k if k <= (data.shape[0] - 2) else (data.shape[0] - 2)
    k = k if (k % 2) else (k - 1)

    data_min = rolling_min(data, window=k, axis=0)
    data_min = data_min + std(data_min, axis=0)

    data_base = rolling_median(data, window=k, axis=0)
    data_base[data_base > data_min] = data_min[data_base > data_min]
    data_base = rolling_mean(data_base, window=k, axis=0)

    data = data - data_base
    data[data < 0] = 0
    data.eliminate_zeros()
    data.sum_duplicates()
    return data


def preprocess(xic: Xic, peak_width: float) -> Xic:
    xic.data = remove_noise(xic.data, peak_width)
    xic.data = remove_baseline(xic.data.tocoo(copy=False), int(10 * peak_width))
    return xic