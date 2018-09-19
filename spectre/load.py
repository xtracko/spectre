import numpy as np

from numba import njit
from spectre.xic import Xic


__all__ = ['from_peaks', 'from_pickle', 'from_mzxml', 'to_pickle']


@njit
def _minmax(values):
    minimum = np.PINF
    maximum = np.NINF

    for value in values:
        minimum = min(minimum, value)
        maximum = max(maximum, value)
    return minimum, maximum


def _mzrange_peaks(spectra):
    minimum = np.PINF
    maximum = np.NINF

    for a, b in (_minmax(mz) for _, mz, _ in spectra):
        minimum = min(minimum, a)
        maximum = max(maximum, b)
    return minimum, maximum


@njit
def _sampling_kernel(transformed_mz, values, nbins: int):
    data = np.zeros(nbins)
    clip = (0 < transformed_mz) & (transformed_mz < nbins)

    for i, value in zip(transformed_mz[clip], values[clip]):
        data[i] = max(value, data[i])
    return data


def _sample_peaks(spectra, nspectra: int, minimum: float, maximum: float, resolution: float):
    beg = int(round(minimum / resolution))
    end = int(round(maximum / resolution))
    nbins = end - beg + 1

    rts = np.zeros(nspectra)
    data = np.zeros((nspectra, nbins))
    mz_scales = np.linspace(minimum, maximum, nbins)

    for i, (rt, mz, values) in enumerate(spectra):
        rts[i] = rt
        data[i] = _sampling_kernel(np.round(mz / resolution).astype(int) - beg, values, nbins)
    return data, rts, mz_scales


def from_peaks(spectra, resolution: float):
    minimum, maximum = _mzrange_peaks(spectra)
    data, rts, mz_scales = _sample_peaks(spectra, len(spectra), minimum, maximum, resolution)
    return Xic(data, rts, mz_scales)


def from_mzxml(file: str, resolution: float):
    from pyopenms import MSExperiment, MzXMLFile

    class Wrapper:
        def __init__(self, exp):
            self.exp = MSExperiment()
            MzXMLFile().load(file.encode(), self.exp)

        def __len__(self):
            return self.exp.getNrSpectra()

        def __iter__(self):
            return ((s.getRT(), *s.get_peaks()) for s in self.exp)

    return from_peaks(Wrapper(file), resolution)


def to_pickle(eic: Xic, file: str):
    from pickle import dump
    dump(eic, open(file, 'wb'))


def from_pickle(file: str):
    from pickle import load
    obj = load(open(file, 'rb'))
    if not isinstance(obj, Xic):
        raise TypeError('The pickle file does not contain a valid Xic class!')
    return obj
