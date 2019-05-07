from typing import Tuple, Union, List

import numpy as np
from scipy.sparse import coo_matrix

from spectre.xic import Xic


def pad(a: coo_matrix, width: Union[int, Tuple, List], mode: str) -> coo_matrix:
    if not isinstance(mode, str):
        raise TypeError('Parameter "mode" must be a string')

    width = np.broadcast_to(width, (a.ndim, 2))
    shape = (a.shape[0] + width[0, 0] + width[0, 1],
             a.shape[1] + width[1, 0] + width[1, 1])
    dtype = (np.promote_types(a.row.dtype, np.min_scalar_type(shape[0])),
             np.promote_types(a.col.dtype, np.min_scalar_type(shape[1])))

    if mode == 'constant':
        coords = (np.add(a.row, width[0, 0], dtype=dtype[0]),
                  np.add(a.col, width[1, 0], dtype=dtype[1]))
        values = a.data.copy()
        return coo_matrix((values, coords), shape, copy=False)

    elif mode == 'symmetric':
        tiles = [[a // size, b // size] for size, (a, b) in zip(a.shape, width)]
        edges = [
            [np.sum((indices < a % size) if ta % 2 else (indices >= -a % size)),
             np.sum((indices >= -b % size) if tb % 2 else (indices < b % size))]
            for (ta, tb), size, indices, (a, b) in
            zip(tiles, a.shape, (a.row, a.col), width)]
        count = np.sum(edges) + np.prod(
            ta + tb + 1 for (ta, tb) in tiles) * a.nnz

        coords = np.empty(count, dtype[0]), np.empty(count, dtype[1])
        values = np.empty(count, a.data.dtype)

        coords[0][:a.row.size] = a.row
        coords[1][:a.col.size] = a.col
        values[:a.data.size] = a.data

        raise NotImplementedError('symmetric mode is incomplete')

    else:
        raise ValueError(f'Unsupported mode "{mode}"')


def rolling_min(a: coo_matrix, k: Tuple[int, ...],
                mode: str = 'valid') -> coo_matrix:
    a = a.tocoo()

    ai, aj, ax = a.row, a.col, a.data
    bi, bj = np.meshgrid(np.arange(k[0]), np.arange(k[1]))

    ci = np.ravel(np.sum(np.meshgrid(ai, bi), axis=0))
    cj = np.ravel(np.sum(np.meshgrid(aj, bj), axis=0))
    cx = np.tile(ax, np.prod(k))

    order = np.lexsort((ci, cj))
    ci, cj, cx = ci[order], cj[order], cx[order]

    unique = np.r_[True, (ci[1:] != ci[:-1]) | (cj[1:] != cj[:-1])]
    indices = np.flatnonzero(unique)
    counts = np.add.reduceat(np.broadcast_to(1, unique.shape), indices)

    ci, cj, cx = ci[unique], cj[unique], np.minimum.reduceat(cx, indices)
    np.minimum(cx, 0, where=counts < np.prod(k), out=cx)

    if mode == 'full':
        return coo_matrix((cx, (ci, cj)), np.add(a.shape, k) - 1)

    elif mode == 'valid':
        shape = np.maximum(a.shape, k) - np.minimum(a.shape, k) + 1

        displacement = -(np.minimum(a.shape, k) - 1)
        np.add(ci, displacement[0], out=ci)
        np.add(cj, displacement[1], out=cj)

        mask = (0 <= ci) & (0 <= cj) & (ci < shape[0]) & (cj < shape[1])
        return coo_matrix((cx[mask], (ci[mask], cj[mask])), shape)

    else:
        raise ValueError(f'Unsupported mode "{mode}"')


def rolling_mean(a: coo_matrix, k: Tuple[int, ...],
                 mode: str = 'valid') -> coo_matrix:
    a = a.tocoo()

    ai, aj, ax = a.row, a.col, a.data
    bi, bj = np.meshgrid(np.arange(k[0]), np.arange(k[1]))

    ci = np.ravel(np.sum(np.meshgrid(ai, bi), axis=0))
    cj = np.ravel(np.sum(np.meshgrid(aj, bj), axis=0))
    cx = np.tile(ax, np.prod(k))

    order = np.lexsort((ci, cj))
    ci, cj, cx = ci[order], cj[order], cx[order]

    unique = np.r_[True, (ci[1:] != ci[:-1]) | (cj[1:] != cj[:-1])]
    indices = np.flatnonzero(unique)
    counts = np.add.reduceat(np.broadcast_to(1, unique.shape), indices)

    ci, cj, cx = ci[unique], cj[unique], np.add.reduceat(cx, indices)
    np.divide(cx, counts, out=cx)

    if mode == 'full':
        return coo_matrix((cx, (ci, cj)), np.add(a.shape, k) - 1)

    elif mode == 'valid':
        shape = np.maximum(a.shape, k) - np.minimum(a.shape, k) + 1

        displacement = -(np.minimum(a.shape, k) - 1)
        np.add(ci, displacement[0], out=ci)
        np.add(cj, displacement[1], out=cj)

        mask = (0 <= ci) & (0 <= cj) & (ci < shape[0]) & (cj < shape[1])
        return coo_matrix((cx[mask], (ci[mask], cj[mask])), shape)

    else:
        raise ValueError(f'Unsupported mode "{mode}"')


def rolling_median(a: coo_matrix, k: Tuple[int, ...],
                   mode: str = 'valid') -> coo_matrix:
    a = a.tocoo()

    ai, aj, ax = a.row, a.col, a.data
    bi, bj = np.meshgrid(np.arange(k[0]), np.arange(k[1]))

    ci = np.ravel(np.sum(np.meshgrid(ai, bi), axis=0))
    cj = np.ravel(np.sum(np.meshgrid(aj, bj), axis=0))
    cx = np.tile(ax, np.prod(k))

    order = np.lexsort((ci, cj, cx))
    ci, cj, cx = ci[order], cj[order], cx[order]

    unique = np.r_[True, (ci[1:] != ci[:-1]) | (cj[1:] != cj[:-1])]
    indices = np.flatnonzero(unique)

    n_negative = np.add.reduceat(cx < 0, indices)
    n_positive = np.add.reduceat(cx > 0, indices)

    mid = np.prod(k) // 2
    neg_mask = mid < n_negative
    pos_mask = mid >= np.prod(k) - n_positive

    result = np.zeros(np.sum(unique), cx.dtype)
    result[neg_mask] = cx[indices[neg_mask] + mid]
    result[pos_mask] = cx[
        (indices + mid - np.prod(k) - n_negative - n_positive)[pos_mask]]

    ci, cj, cx = ci[unique], cj[unique], result

    if mode == 'full':
        return coo_matrix((cx, (ci, cj)), np.add(a.shape, k) - 1)

    elif mode == 'valid':
        shape = np.maximum(a.shape, k) - np.minimum(a.shape, k) + 1

        displacement = -(np.minimum(a.shape, k) - 1)
        np.add(ci, displacement[0], out=ci)
        np.add(cj, displacement[1], out=cj)

        mask = (0 <= ci) & (0 <= cj) & (ci < shape[0]) & (cj < shape[1])
        return coo_matrix((cx[mask], (ci[mask], cj[mask])), shape)

    else:
        raise ValueError(f'Unsupported mode "{mode}"')


def std(a: coo_matrix, axis: int) -> coo_matrix:
    if not isinstance(a, coo_matrix):
        raise TypeError("Matrix is not in COO format")
    if axis not in (0, 1):
        raise ValueError(
            f"Unsupported axis value {axis} for 2 dimensional matrix")

    if not a.has_canonical_format:
        a.sum_duplicates()

    (major, minor), data = ((a.row, a.col) if axis else (a.col, a.row)), a.data
    major, minor, data = np.copy(major), np.copy(minor), np.copy(data)

    order = np.lexsort((major, minor))
    major, data = major[order], data[order]

    unique = np.flatnonzero(np.r_[True, major[1:] != major[:-1]])
    major, minor = major[unique], np.zeros_like(minor)

    mean = np.add.reduceat(data, unique) / a.shape[axis]
    var = (np.add.reduceat(data * data, unique) / a.shape[axis]) - (mean * mean)

    coords = (major, minor) if axis else (minor, major)
    shape = (a.shape[0], 1) if axis else (1, a.shape[1])
    return coo_matrix((np.sqrt(var), coords), shape, copy=False)


def remove_baseline(data: coo_matrix, k: int) -> coo_matrix:
    k = k if k <= (data.shape[0] - 2) else (data.shape[0] - 2)
    k = k if (k % 2) else (k - 1)

    data_pad = pad(data, ((k // 2, k // 2), (0, 0)), mode='constant')
    data_min = rolling_min(data_pad, (k, 1))
    data_min = data_min + np.squeeze(std(data_min, axis=0).toarray())

    data_base = rolling_median(data_pad, (k, 1))
    data_base, data_min = data_base.tocsr(False), data_min
    data_base[data_base > data_min] = data_min[data_base > data_min]
    data_base = data_base.tocoo(False)

    data_base = pad(data_base, ((k // 2, k // 2), (0, 0)), mode='constant')
    data_base = rolling_mean(data_base, (k, 1))

    data = data - data_base
    data[data < 0] = 0
    data.eliminate_zeros()
    data.sum_duplicates()
    return data


def preprocess(xic: Xic, peak_width: float) -> Xic:
    from spectre.sparse.preprocess_naive import remove_noise
    xic.data = remove_noise(xic.data, peak_width)
    xic.data = remove_baseline(xic.data.tocoo(copy=False), int(10 * peak_width))
    return xic
