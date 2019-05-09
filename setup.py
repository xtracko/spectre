from skbuild import setup
from setuptools import find_packages

setup(
    name='spectre',
    version='0.0.1',
    description='Sparse spectral deconvolution of hi-res scans from GS/MS',
    author='xtracko',
    author_email='novotng@gmail.com',
    url='https://github.com/xtracko/spectre',
    packages=find_packages(exclude=['tests', 'benchmarks']),
    zip_safe=False,
    setup_requires=['numpy'],
    install_requires=['numba', 'numpy', 'scipy', 'pyopenms'],
    tests_require=['pytest', 'pytest-benchmark'],
    test_suite='tests'
)
