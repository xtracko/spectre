#pragma once

#include <tuple>
#include <type_traits>

namespace xs {
  struct coo_array_tag {};
  struct csr_array_tag {};
  struct csc_array_tag {};

  template <typename Array>
  constexpr inline bool is_coo_array =
      std::is_base_of_v<coo_array_tag, std::decay_t<Array>>;

  template <typename Array>
  constexpr inline bool is_csr_array =
      std::is_base_of_v<csr_array_tag, std::decay_t<Array>>;

  template <typename Array>
  constexpr inline bool is_csc_array =
      std::is_base_of_v<csc_array_tag, std::decay_t<Array>>;
} // namespace xs

namespace xs {
  template <typename Array>
  using array_value_t = typename std::decay_t<Array>::value_type;

  template <typename Array, auto Axis>
  using array_coord_t = typename std::decay_t<Array>::template coord_type<Axis>;
} // namespace xs