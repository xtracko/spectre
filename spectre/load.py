from os import fsencode
from typing import Union, Text, Any

import numpy as np

from spectre.spectra import sample_spectra
from spectre.xic import Xic


def from_spectra(scans: np.ndarray, peaks: np.ndarray, values: np.ndarray,
                 retention_times: np.ndarray, sampling: float) -> Xic:
    """Create a spectre project from sparse spectral data.

    Args:
        scans:
        peaks:
        values:
        retention_times:
        sampling: A sampling resolution

    Returns:
        Xic: A Spectre project.

    Raises:
        ValueError:
    """
    if scans[0] != 0 or np.any(scans[:-1] > scans[1:]) \
            or scans[-1] != len(peaks):
        raise ValueError('Scans is not a valid pointer array to the peaks '
                         'array!')

    if not np.all(retention_times[:-1] < retention_times[1:]):
        raise ValueError('Scans are not sorted by retention times!')

    sampled = sample_spectra(scans, peaks, values, sampling)

    min_mz = np.round(peaks.min() * (1/sampling)) * sampling
    max_mz = np.round(peaks.max() * (1/sampling)) * sampling

    return Xic(data=sampled,
               mz_scales=np.linspace(min_mz, max_mz, sampled.shape[1], True),
               rt_scales=retention_times)


def from_mzxml(file: Union[Text, Any], sampling: float) -> Xic:
    """Create a Spectre project from a mzXML file.

    Args:
        file (Union[Text, PathLike]): A valid path to the mzXML file.
        sampling (float): A sampling resolution (see :func:`from_spectra`).

    Returns:
        Xic: A Spectre project.
    """
    from pyopenms import MSExperiment, MzXMLFile

    exp = MSExperiment()
    MzXMLFile().load(fsencode(file), exp)

    if not exp.isSorted():
        exp.sortSpectra()

    scans = np.r_[0, np.cumsum([scan.size() for scan in exp])]
    peaks = np.empty(scans[-1], np.float64)
    values = np.empty(scans[-1], np.float64)
    retention_times = np.asarray([scan.getRT() for scan in exp])

    for a, b, spectrum in zip(scans[:-1], scans[1:], exp):
        peaks[a:b], values[a:b] = spectrum.get_peaks()
    return from_spectra(scans, peaks, values, retention_times, sampling)


def from_pickle(file: Union[Text, Any]) -> Xic:
    """Load a Spectre project from a pickle file.

    The saving/loading of a pickle file is much more faster then parsing
    mzXML format or other traditional formats.

    Args:
        file (Union[Text, PathLike]): A valid path to the pickle file.

    Returns:
        Xic: A Spectre project.

    Raises:
        TypeError: The file does not contain a valid Spectre project.
        OSError: An error occurred during reading the file.
    """
    import pickle

    with open(file, 'rb') as f:
        obj = pickle.load(f)

    if not isinstance(obj, Xic):
        raise TypeError('The file is not a valid Spectre project!')
    return obj


def to_pickle(obj: Xic, file: Union[Text, Any]) -> None:
    """Save a Spectre project as a pickle file.

    The saving/loading of a pickle file is much more faster then parsing
    mzXML format or other traditional formats.

    Args:
        obj (Xic): A Spectre project.
        file (Union[Text, PathLike]): Path to the file.

    Raises:
        TypeError: The object to pickle is not a valid Spectre project.
        OSError: An error occurred during writing to the file.
    """
    import pickle

    if not isinstance(obj, Xic):
        raise TypeError('The supplied object is not a valid Spectre project!')

    with open(file, 'wb') as f:
        pickle.dump(obj, f)
