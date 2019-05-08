# Spectre
Deconvolution of hi-res GS/MS spectra

## Build
### Requirements
  * [python](https://www.python.org/) version 3.5 or higher
  * [cmake](https://cmake.org/) version 3.14 or higher
  * [gcc](https://gcc.gnu.org/) or [clang](https://clang.llvm.org/) with min c++17 supported and with proper version of libstdc++/libc++
  * [pybind11](https://github.com/pybind/pybind11) version 2.2.4
  * [scikit-build](https://github.com/scikit-build/scikit-build) version 0.8 or higher
  * [pyopenms](https://pyopenms.readthedocs.io) version 2.2.4
  * [numpy](http://www.numpy.org/) version 1.15 or higher
  * [scipy](http://www.scipy.org/) version 1.2 or higher
  
### Native build
To make native build of Spectre from sources install the requirements (see above) via your system package manager, conda, vcpkg or your preffered package manager. You can choose from installing Spectre locally, building a wheel, or installing Spectre in developer mode inside the the source directory (please refer to the documentation of [scikit-build](https://github.com/scikit-build/scikit-build) and [setuptools](https://setuptools.readthedocs.io/)).

 ```
  git clone https://github.com/xtracko/spectre.git
  cd spectre
  ```
Then choose one of the following options to build and/or install:
  ```
  python -m setup install -- -Dpybind11_DIR=/path/to/pybind11
  python -m setup bdist_wheel -- -Dpybind11_DIR=/path/to/pybind11
  python -m setup develop --build-type [Debug|Release] -- -Dpybind11_DIR=/path/to/pybind11
  ```
If you are having troble with inconsistent python versions (especially when developing C code in IDE) please pass both folloving variable to cmake:
  ```
  -DPYTHON_EXECUTABLE=/path/to/your/python/executable
  -DPython_ROOT_DIR=/path/to/your/python/root/dir
  ```
  
### Containerized build
You can also build Spectre for Debian 9 (Stretch) and python-3.5 (current python3 for Debian 9) using a [Singularity](https://www.sylabs.io/) container.

Build the Singularity container first (needs a machine with root access):

  ```bash
  git clone https://github.com/xtracko/spectre.git
  cd spectre/tools/spectre_build
  sudo singularity build spectre_build.sif spectre_build.def
  ```
Then on you target machine with singularity installed (no need for root access from now on):
  ```bash
  # continuing the build from the directory spectre/tools/spectre_build
  singularity exec spectre_build.sif /bin/bash
  cd ../..
  # if you have conda installed on your system diable it
  conda deactivate
  # build a wheel (no need for aditional arguments as in the native build)
  python -m setup bdist_wheel
  ```
