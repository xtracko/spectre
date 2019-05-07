cmake_minimum_required(VERSION 3.14)
project(spectre)


find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 2.2.4 REQUIRED)


add_library(_sparse MODULE
        spectre/sparse/src/_sparse.cpp)
target_compile_features(_sparse PUBLIC
        cxx_std_17)
target_compile_definitions(_sparse PRIVATE
        NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION)
target_link_libraries(_sparse PRIVATE
        pybind11::module
        Python::NumPy)
set_target_properties(_sparse PROPERTIES
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
        INTERPROCEDURAL_OPTIMIZATION TRUE)


install(TARGETS _sparse LIBRARY
        DESTINATION spectre/sparse)