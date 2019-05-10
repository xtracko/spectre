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
        raise TypeError("unsupported type given '{}'".format(type(x)))


def std(x: Union[csr_matrix, csc_matrix], axis: int):
    if axis not in (0, 1):
        raise ValueError(
            "Unsupported axis value {} for 2 dimensional matrix".format(axis))

    x = x.tocsc() if axis == 0 else x.tocsr()
    y = np.zeros(shape=x.shape[1 - axis], dtype=x.dtype)

    _sparse.std_csr(x.indptr, x.data, x.shape[axis], y)
    return y


def rolling(func, x: Union[csr_matrix, csc_matrix], window: int, axis: int):
    x = x.tocsr() if axis else x.tocsc()
    x.sum_duplicates()
    minor_size = x.shape[1] if axis else x.shape[0]

    needs_64bit = min(np.prod(x.shape), x.nnz * window) > np.iinfo(np.int32).max
    index_type = np.int64 if needs_64bit else x.indptr.dtype

    pointers = np.empty(x.indptr.shape, dtype=index_type)
    size = _sparse.rolling_alloc_csr(x.indptr, x.indices, pointers, minor_size,
                                     window)
    data = np.zeros(size, dtype=x.data.dtype)
    indices = np.zeros(size, dtype=index_type)

    func(x.indptr, x.indices, x.data, indices, data, minor_size, window)

    if axis:
        return csr_matrix((data, indices, pointers), x.shape)
    else:
        return csc_matrix((data, indices, pointers), x.shape)


def rolling_min(x: Union[csr_matrix, csc_matrix], window: int, axis: int):
    return rolling(_sparse.rolling_min_csr, x, window, axis)


def rolling_max(x: Union[csr_matrix, csc_matrix], window: int, axis: int):
    return rolling(_sparse.rolling_max_csr, x, window, axis)


def rolling_mean(x: Union[csr_matrix, csc_matrix], window: int, axis: int):
    return rolling(_sparse.rolling_mean_csr, x, window, axis)


def rolling_median(x: Union[csr_matrix, csc_matrix], window: int, axis: int):
    return rolling(_sparse.rolling_median_csr, x, window, axis)


def max_clip_spmat_plus_dvec(a_mat: Union[csr_matrix, csc_matrix],
                             b_mat: Union[csr_matrix, csc_matrix],
                             c_vec: np.ndarray):
    assert (a_mat.shape == b_mat.shape and c_vec.ndim == 1)
    assert ((isspmatrix_csr(a_mat) and isspmatrix_csr(b_mat)) or (
            isspmatrix_csc(a_mat) and isspmatrix_csc(b_mat)))
    assert ((isspmatrix_csr(a_mat) and a_mat.shape[0] == c_vec.shape[0]) or (
            isspmatrix_csc(a_mat) and a_mat.shape[1] == c_vec.shape[0]))

    # compute a[a > b + c] = (b + c)[a > b + c]
    _sparse.maxclip_csr_spmat_plus_dvec_nonnegative(a_mat.indptr, a_mat.indices,
                                                    a_mat.data, b_mat.indptr,
                                                    b_mat.indices, b_mat.data,
                                                    c_vec)


def savgol_filter(x: Union[csr_matrix, csc_matrix], window: int, degree: int,
                  axis: int):
    x = x.tocsr() if axis else x.tocsc()
    x.sum_duplicates()

    minor_size = x.shape[1] if axis else x.shape[0]

    needs_64bit = min(np.prod(x.shape), x.nnz * window) > np.iinfo(np.int32).max
    index_type = np.int64 if needs_64bit else x.indptr.dtype

    pointers = np.empty(x.indptr.shape, dtype=index_type)
    size = _sparse.rolling_alloc_csr(x.indptr, x.indices, pointers, minor_size,
                                     window)
    data = np.zeros(size, dtype=x.data.dtype)
    indices = np.zeros(size, dtype=index_type)

    from scipy.signal import savgol_coeffs
    coeffs = savgol_coeffs(window, degree)
    _sparse.convolve_csr_dv(x.indptr, x.indices, x.data, coeffs, indices, data,
                            minor_size)

    if axis:
        return csr_matrix((data, indices, pointers), x.shape)
    else:
        return csc_matrix((data, indices, pointers), x.shape)


def remove_noise(data: Union[csr_matrix, csc_matrix], peak_width: float):
    peak_width = int(round(peak_width))
    window = peak_width if (peak_width % 2) else (peak_width + 1)
    degree = min(3, window - 1)

    if degree > 0:
        data = savgol_filter(data, window, degree, axis=0)

    # IDEA: implement clipping of items <0 and then pruning of an array in
    # 1-pass algorithm (currently at least 3 pass)
    data[data < 0] = 0
    data.eliminate_zeros()
    data.sum_duplicates()
    return data


def remove_baseline(data: Union[csr_matrix, csc_matrix], k: int):
    k_med = min(k, data.shape[0] - 1)
    k_med = k_med if (k_med % 2) else (k_med - 1)

    data_min = rolling_min(data, window=k, axis=0)

    data_base = rolling_median(data, window=k_med, axis=0)
    max_clip_spmat_plus_dvec(data_base, data_min, std(data_min, axis=0))
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
