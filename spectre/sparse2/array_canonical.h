#pragma once

#include "array_coo.h"
#include <algorithm>
#include <iterator>

template <unsigned... Order> struct coordinate_comparator {
  template <typename A, typename B>
  constexpr bool operator()(A&& a, B&& b) const {
    auto ordered_a = std::tuple{std::get<Order>(std::forward<A>(a))...};
    auto ordered_b = std::tuple{std::get<Order>(std::forward<B>(b))...};

    return ordered_a < ordered_b;
  }
};

template <typename Array, typename Comparator>
auto is_canonical(Array&& array, Comparator&& comparator)
    -> std::enable_if_t<is_coo_array_v<Array>, bool> {
  if (array.empty())
    return true;

  bool result = true;
  const auto coords = array.coords().first(array.size() - 1);

#pragma omp parallel for simd reduction(&& : result) schedule(static, 512)
  for (auto iter = std::cbegin(coords); iter < std::cend(coords); ++iter) {
    result = result && comparator(iter[0], iter[1]);
  }
  return result;
}

template <typename Array, typename Comparator>
auto to_canonical(Array&& array, Comparator&& comparator)
    -> std::enable_if_t<is_coo_array_v<Array>, Array> {
  if (is_canonical(array, comparator))
    return std::forward<Array>(array);

  std::sort(std::begin(array.sparse()), std::end(array.sparse()),
            [=](auto&& a, auto&& b) {
              return comparator(std::get<0>(a), std::get<0>(b));
            });
  return std::forward<Array>(array);
}

using row_major_order = coordinate_comparator<0, 1>;
using col_major_order = coordinate_comparator<1, 0>;
