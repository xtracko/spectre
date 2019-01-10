#include <benchmark/benchmark.h>

#include <algorithm>
#include <iterator>
#include <random>
#include <vector>

template <typename Iter> static void radix_sort(const Iter beg, const Iter end) {
  using value_type = typename std::iterator_traits<Iter>::value_type;

  for (value_type mask = 1; mask; mask <<= 1) {
    std::stable_partition(beg, end, [mask](auto val) { return !(val & mask); });
  }
}

template <typename Type> static void bench_std_sort(benchmark::State &state) {
  std::vector<Type> data(static_cast<std::size_t>(state.range(0)));

  std::random_device rd;
  std::default_random_engine gen{rd()};
  std::uniform_int_distribution<Type> dis;

  for (auto _ : state) {
    state.PauseTiming();
    std::generate(data.begin(), data.end(),
                  [&gen, &dis]() { return dis(gen); });
    state.ResumeTiming();

    std::sort(data.begin(), data.end());
  }
}

template <typename Type> static void bench_radix_sort(benchmark::State &state) {
  std::vector<Type> data(static_cast<std::size_t>(state.range(0)));

  std::random_device rd;
  std::default_random_engine gen{rd()};
  std::uniform_int_distribution<Type> dis;

  for (auto _ : state) {
    state.PauseTiming();
    std::generate(data.begin(), data.end(),
                  [&gen, &dis]() { return dis(gen); });
    state.ResumeTiming();

    std::stable_sort(data.begin(), data.end());
  }
}

const auto beg = 1024;
const auto end = 1024 << 10;

BENCHMARK_TEMPLATE(bench_std_sort, std::uint64_t)->Range(beg, end);
BENCHMARK_TEMPLATE(bench_radix_sort, std::uint64_t)->Range(beg, end);