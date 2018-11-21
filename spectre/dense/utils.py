import numpy as np


def _rolling_window(a: np.ndarray, k: int, axis: int = 0) -> np.ndarray:
    shape = a.shape[:axis] + (a.shape[axis] - k + 1, k) + a.shape[(axis + 1):]
    strides = a.strides[:(axis + 1)] + (a.strides[axis],) + a.strides[(axis + 1):]
    return np.lib.stride_tricks.as_strided(a, shape, strides)


def rolling_min(a: np.ndarray, k: int, axis: int = 0) -> np.ndarray:
    return np.nanmin(_rolling_window(a, k, axis), axis=axis + 1)


def rolling_max(a: np.ndarray, k: int, axis: int = 0) -> np.ndarray:
    return np.nanmax(_rolling_window(a, k, axis), axis=axis + 1)


def rolling_mean(a: np.ndarray, k: int, axis: int = 0) -> np.ndarray:
    return np.mean(_rolling_window(a, k, axis), axis=axis + 1)


def rolling_median(a: np.ndarray, k: int, axis: int = 0) -> np.ndarray:
    return np.median(_rolling_window(a, k, axis), axis=axis + 1)
