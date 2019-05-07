from typing import Tuple

import numpy as np
from numba import njit, prange


@njit
def is_csr_indexing(rows: np.ndarray, cols: np.ndarray) -> bool:
    """Check whether a set of arrays satisfies the requirements for CSR
    indexing.

    The requirements for CSR (compressed sparse row) indexing are:

    * :code:`rows.ndim == 1`
    * :code:`cols.ndim == 1`
    * :code:`rows.size > 0`
    * :code:`rows[0] >= 0`
    * :code:`rows[:-1] <= rows[1:]`
    * :code:`rows[-1] <= len(cols)`

    Args:
        rows (numpy.ndarray): An array of row indices pointing to the
            beginning and end of each column.
        cols (numpy.ndarray): An integer array of sparse columns indices.

    Returns:
        bool: True if the input arrays satisfies the CSR indexing requirements.
    """
    return rows.ndim == cols.ndim == 1 and rows.size > 0 and rows[
        0] >= 0 and np.all(rows[:-1] <= rows[1:]) and rows[-1] <= cols.size


@njit
def is_csr_matrix(rows: np.ndarray, cols: np.ndarray, data: np.ndarray) -> bool:
    """Check whether a set of arrays satisfies the requirements for CSR matrix.

    The requirements for CSR (compressed sparse row) matrix are:

    * all the requirements for :func:`is_csr_indexing`
    * :code:`data.shape == cols.shape`

    Args:
        rows (numpy.ndarray): An array of row indices pointing to the
            beginning and end of each column.
        cols (numpy.ndarray): An integer array of sparse columns indices.
        data (Optional[numpy.ndarray]): An array containing sparse values.

    Returns:
        bool: True if the input arrays satisfies the CSR matrix requirements.
    """
    return is_csr_indexing(rows, cols) and cols.shape == data.shape


@njit(parallel=True, nogil=True)
def is_sorted_by_cols(rows: np.ndarray, cols: np.ndarray) -> bool:
    """Check whether CSR sparse column indices are sorted within each row.

    Args:
        rows (numpy.ndarray): An array of row indices in CSR format.
        cols (numpy.ndarray): An array of sparse column indices in CSR format.

    Returns:
        bool: True if column indices are sorted.
    """
    assert is_csr_indexing(rows, cols)

    non_descending = True

    for i in prange(rows.size - 1):
        a, b = rows[i], rows[i + 1]
        non_descending &= np.all(cols[a:(b - 1)] <= cols[(a + 1):b])
    return non_descending


@njit(parallel=True, nogil=True)
def sort_cols(rows: np.ndarray, cols: np.ndarray, data: np.ndarray):
    """An inplace sort of column indices within each row.

    Args:
        rows (numpy.ndarray): An array of row indices in CSR format.
        cols (numpy.ndarray): An array of sparse column indices in CSR format.
        data (numpy.ndarray): An array of sparse data values in CSR format.
    """
    assert is_csr_matrix(rows, cols, data)

    for i in prange(rows.size - 1):
        a, b = rows[i], rows[i + 1]

        if not np.all(cols[a:b - 1] <= cols[a + 1:b]):
            indices = np.argsort(cols[a:b])
            cols[a:b] = cols[a:b][indices]
            data[a:b] = data[a:b][indices]


@njit
def unique_cols(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Find unique columns in each row.

    Args:
        rows (numpy.ndarray): An array of row indices in CSR format.
        cols (numpy.ndarray): An array of sparse columns indices
            sorted within each row (see :func:`is_sorted_by_cols` and
            :func:`sort_by_cols`).

    Returns:
        numpy.ndarray: Returns a boolean array of same length as 'cols' where
        the truth value at `i`-th position denotes whether the index at
        `cols[i]` is unique within its own row, or not.

    Notes:
        Columns must be sorted (use :func:`sort_by_cols`).
    """
    assert is_sorted_by_cols(rows, cols)

    mask = np.ones_like(cols, np.bool_)

    for i in range(rows.size - 1):
        a, b = rows[i], rows[i + 1]
        mask[a + 1:b] = cols[a:b - 1] != cols[a + 1:b]
    return mask


@njit(parallel=True, nogil=True)
def merge_cols(rows: np.ndarray, cols: np.ndarray, data: np.ndarray, func) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge non-unique column indices within each row.

    Perform an reduction operation on values of non-unique column indices
    within each row.

    Args:
        rows (numpy.ndarray): An array of row indices in CSR format.
        cols (numpy.ndarray): An array of sparse column indices in CSR format
            sorted within each row (see :func:`is_sorted_by_cols` and
            :func:`sort_by_cols`).
        data (numpy.ndarray): An array of sparse data values in CSR format
            corresponding to the sorted column indices.
        func (callable): A reduction operation to perform on data with
            non-unique column indices within it's own row.

    Returns:
        (numpy.ndarray, numpy.ndarray, numpy.ndarray): CSR matrix with unique
        column indices within each row, created by merging the values of
        non-unique column indices.
    """
    assert is_csr_matrix(rows, cols, data)
    assert is_sorted_by_cols(rows, cols)

    unique_mask = unique_cols(rows, cols)

    rows_out = np.zeros_like(rows)
    for i in prange(rows.size - 1):
        rows_out[i + 1] = np.sum(unique_mask[rows[i]:rows[i + 1]])
    for i in range(1, rows_out.size):
        rows_out[i] = rows_out[i - 1] + rows_out[i]

    cols_out = np.empty(rows_out[-1], cols.dtype)
    data_out = np.empty(rows_out[-1], data.dtype)

    for i in prange(rows.size - 1):
        a = rows[i]

        for j in range(rows_out[i], rows_out[i + 1]):
            b = a + 1
            while b != rows[i + 1] and not unique_mask[b]:
                b += 1

            cols_out[j] = cols[a]
            data_out[j] = func(data[a:b])
            a = b

    return rows_out, cols_out, data_out
