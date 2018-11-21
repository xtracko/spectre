import numpy as np

from scipy import signal


def gaussian_kernel(peak_width, cutoff=0.01):
    # expression of the standard deviation from definition of an gaussian window
    # `w(n) = e ^ -0.5 * (n / std)^2`, where `n` is `peak_width / 2`, taken from
    # https://docs.scipy.org/doc/scipy-1.1.0/reference/generated/scipy.signal.windows.gaussian.html
    std = peak_width / np.sqrt(-8 * np.log(cutoff))
    return signal.gaussian(np.round(peak_width), std)


def match_filter(data, kernel):
    kernel_width = kernel.shape[0]

    ranks = np.zeros_like(data)
    norms = np.zeros_like(data)
    conds = np.zeros_like(data)

    for i in data.shape[0]:
        di = data[i:kernel_width]
        ci = np.cov(di, rowvar=False)

        ranks[i] = np.linalg.matrix_rank(ci)
        norms[i] = np.linalg.norm(ci)
        conds[i] = np.linalg.cond(ci)

    return ranks, norms, conds
