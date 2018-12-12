cmake_minimum_required(VERSION 3.10)
project(xsparse)


set(PYBIND11_CPP_STANDARD -std=c++17)


find_package(pybind11 2.2.4 REQUIRED)
find_package(GTest 1.8)
find_package(benchmark 1.4)


add_library(xsparse INTERFACE)
target_include_directories(xsparse INTERFACE xsparse/)
target_compile_features(xsparse INTERFACE cxx_std_17)


if (GTest_FOUND)
    add_executable(xsparse-test
            xsparse-test/test_coo.cpp
            xsparse-test/test_span.cpp
            xsparse-test/test_zip.cpp)
    target_link_libraries(xsparse-test xsparse GTest::GTest GTest::Main)

    include(GoogleTest)
    gtest_discover_tests(xsparse-test)
endif (GTest_FOUND)


if (benchmark_FOUND)
    add_executable(xsparse-bench xsparse-bench/main.cpp)
    target_link_libraries(xsparse-bench PRIVATE benchmark::benchmark benchmark::main)
endif (benchmark_FOUND)


pybind11_add_module(pysparse spectre/xsparse/xsparse.cpp)
target_link_libraries(pysparse PRIVATE xsparse)