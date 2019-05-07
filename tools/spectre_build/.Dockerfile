FROM gcc:latest

# install python3.5, git and tar
RUN set -ex \
    && apt-get update && apt-get install -y \
    git tar python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# install cmake version 3.14
RUN set -ex \
    && wget -qO- 'https://cmake.org/files/v3.14/cmake-3.14.0-Linux-x86_64.tar.gz' | tar --strip-components=1 -xz -C /usr/local

WORKDIR /workdir

# using pip install python dependencies
RUN set -ex \
    && pip3 install numpy numba pytest scikit-build scipy pyopenms

# install pybind11 version 2.2.4
RUN set -ex \
    && wget -qO- 'https://github.com/pybind/pybind11/archive/v2.2.4.tar.gz' | tar -xz \
    && mkdir buid-pybind11 \
    && cd buid-pybind11 \
    && cmake ../pybind11-2.2.4 \
    && make install -j \
    && rm -rf ../pybind11-2.2.4 ../build-pybind11
