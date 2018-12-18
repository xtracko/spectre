cmake_minimum_required(VERSION 3.10)
project(xsparse)


option(BUILD_Tests "Build tests" ON)
option(BUILD_Benchmarks "Build benchmarks" ON)


add_subdirectory(xsparse)


set(PYBIND11_CPP_STANDARD -std=c++17)
find_package(pybind11 2.2.4 REQUIRED)

pybind11_add_module(pysparse spectre/xsparse/xsparse.cpp)
target_link_libraries(pysparse PRIVATE xsparse)