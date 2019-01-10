find_package(TBB 2019.0 REQUIRED)

add_library(xsparse INTERFACE)
target_compile_features(xsparse INTERFACE cxx_std_17)
target_include_directories(xsparse INTERFACE include)
target_link_libraries(xsparse INTERFACE ${TBB_IMPORTED_TARGETS})


if(BUILD_Tests)
    find_package(GTest 1.8 REQUIRED)

    add_executable(xsparse-test
            tests/test_zip.cpp)
    target_link_libraries(xsparse-test
            xsparse
            GTest::GTest
            GTest::Main)

    enable_testing()
    include(GoogleTest)
    gtest_discover_tests(xsparse-test)
endif(BUILD_Tests)


if(BUILD_Benchmarks)
    find_package(benchmark REQUIRED)

    add_executable(xsparse-benchmark
            benchmarks/bench_radix_sort.cpp)
    target_link_libraries(xsparse-benchmark PRIVATE
            xsparse
            benchmark::benchmark
            benchmark::benchmark_main)
endif(BUILD_Benchmarks)