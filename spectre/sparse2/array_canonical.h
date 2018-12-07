#pragma once

#include "array_view.h"
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
auto is_canonical(Array const& array, Comparator&& comparator)
    -> std::enable_if_t<is_coo_array_v<Array>, bool> {
  if (array.empty())
    return true;

  bool result = true;
  auto axis1 = coords<1>(array).cbegin();
  const auto sentinel = coords<0>(array).cend() - 1;

#pragma omp parallel for simd reduction(&& : result) linear(axis1 : 1) schedule(static, 512)
  for (auto axis0 = coords<0>(array).cbegin(); axis0 < sentinel; ++axis0) {
    result &= comparator(std::tuple{axis0[0], axis1[0]},
                         std::tuple{axis0[1], axis1[1]});
  }
  return result;
}

template <typename Array, typename Comparator>
auto to_canonical(Array&& array, Comparator&& comparator)
    -> std::enable_if_t<!std::is_const_v<Array> && is_coo_array_v<Array>,
                        Array> {
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