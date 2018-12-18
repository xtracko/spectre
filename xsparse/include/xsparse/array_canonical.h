#pragma once

#include <array_traits.h>
#include <type_traits>

namespace xsparse {
  template <typename Array>
  std::enable_if_t<is_coo_array_v<Array>, bool>
  is_canonical(const Array &array) {
    using std::begin;
    using std::end;

    auto &&coords = array.coords();
    return std::is_sorted(begin(coords), end(coords));
  }

  template <typename Array>
  std::enable_if_t<is_coo_array_v<Array>, std::add_rvalue_reference_t<Array>>
  to_canonical(Array &&array) {
    using std::begin;
    using std::end;

    std::sort(begin(array), end(array), [](const auto &lhs, const auto &rhs) {
      return std::get<0>(lhs) < std::get<0>(rhs);
    });

    return std::forward<Array>(array);
  }
} // namespace xsparse