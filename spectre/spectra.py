import numpy as np
from numba import njit
from scipy.sparse import csr_matrix

from spectre.sparse.indexing import merge_cols
from spectre.sparse.indexing import sort_cols


@njit
def _max_poll(array):
    return np.nanmax(array)


def sample_spectra(spectra: np.ndarray, peaks: np.ndarray, values: np.ndarray, sampling: float) -> csr_matrix:
    rows = spectra

    cols = np.multiply(peaks, 1 / sampling)
    cols = np.round(cols, out=cols).astype(np.int64, copy=False)
    cols = np.subtract(cols, cols.min(), out=cols)

    data = values.astype(np.min_scalar_type(values))

    sort_cols(rows, cols, data)
    rows, cols, data = merge_cols(rows, cols, data, _max_poll)

    return csr_matrix((data, cols, rows), dtype=data.dtype)
