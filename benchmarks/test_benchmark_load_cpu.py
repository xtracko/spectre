import os

import pytest

import spectre

file_list = 'CalNFR1.mzXML'
precision_list = [1, 0.1, 0.01, 0.01, 0.001]


@pytest.mark.parametrize("file", file_list)
@pytest.mark.parametrize("precision", precision_list)
def test_bench_from_mzxml(benchmark, file, precision):
    benchmark(spectre.from_mzxml, file, precision)


@pytest.mark.parametrize("file", file_list)
@pytest.mark.parametrize("precision", precision_list)
def test_bench_from_pickle(benchmark, file, precision):
    pickle_file = '{}.pickle'.format(file)

    assert not os.path.exists(pickle_file)

    spectre.from_mzxml(file, precision).to_pickle(pickle_file)
    benchmark(spectre.from_pickle(file + '.pickle'))
    os.remove(pickle_file)
