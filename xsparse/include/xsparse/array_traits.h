#pragma once

#include <tuple>
#include <type_traits>

namespace xsparse {
  struct coo_array_tag {};
  struct csr_array_tag {};
  struct csc_array_tag {};

  template <typename Array>
  inline constexpr bool
      is_coo_array_v = std::is_base_of_v<coo_array_tag, std::decay_t<Array>>;

  template <typename Array>
  inline constexpr bool
      is_csr_array_v = std::is_base_of_v<csr_array_tag, std::decay_t<Array>>;

  template <typename Array>
  inline constexpr bool
      is_csc_array_v = std::is_base_of_v<csc_array_tag, std::decay_t<Array>>;

  template <typename Array>
  using array_shape_t = typename std::decay_t<Array>::shape_type;

  template <typename Array>
  using array_value_t = typename std::decay_t<Array>::value_type;
} // namespace xsparse