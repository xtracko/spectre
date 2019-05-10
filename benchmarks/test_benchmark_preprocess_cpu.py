import pytest

import spectre

file_list = ['CalNFR1.mzXML']
precision_list = [1, 0.1, 0.01, 0.001]


@pytest.mark.parametrize("file", file_list)
@pytest.mark.parametrize("precision", precision_list)
def test_bench_naive_preprocess(benchmark, file, precision):
    experiment = spectre.from_mzxml(file, precision)

    def function(ex):
        spectre.naive.preprocess(xic=ex.copy(), peak_width=5)
    benchmark(function, experiment)


@pytest.mark.parametrize("file", file_list)
@pytest.mark.parametrize("precision", [1])
def test_bench_numpy_preprocess(benchmark, file, precision):
    experiment = spectre.from_mzxml(file, precision)

    def function(ex):
        spectre.numpy.preprocess(xic=ex.copy(), peak_width=5)
    benchmark(function, experiment)


@pytest.mark.parametrize("file", file_list)
@pytest.mark.parametrize("precision", precision_list)
def test_bench_cpp_preprocess(benchmark, file, precision):
    experiment = spectre.from_mzxml(file, precision)

    def function(ex):
        spectre.cpp.preprocess(xic=ex.copy(), peak_width=5)
    benchmark(function, experiment)
