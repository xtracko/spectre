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

  template <std::size_t Axis, typename Array, typename = std::void_t<>>
  struct coord_type;

  template <std::size_t Axis, typename Array>
  struct coord_type<Axis, Array, std::enable_if_t<is_coo_array_v<Array>>> {
    using type = std::tuple_element_t<Axis, typename Array::coord_type>;
  };

  template <std::size_t Axis, typename Array>
  using coord_type_t = typename coord_type<Axis, std::decay_t<Array>>::type;
} // namespace xsparse