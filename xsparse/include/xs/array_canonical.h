#pragma once

#include <algorithm>
#include <xs/array_traits.h>

namespace xs {
  template <typename Array>
  std::enable_if_t<is_coo_array<Array>, bool> is_canonical(const Array &array) {
    using std::begin;
    using std::end;

    auto&& indices = array.indices();
    std::is_sorted(begin(indices), end(indices));
  }

  template <typename Array>
  std::enable_if_t<is_coo_array<Array>, std::add_rvalue_reference_t<Array>>
  to_canonical(Array &&array) {
    using std::begin;
    using std::end;

    std::sort(begin(array), end(array), [](const auto &lhs, const auto &rhs) {
      return std::get<0>(lhs) < std::get<0>(rhs);
    });

    return std::forward<Array>(array);
  }
} // namespace xs




