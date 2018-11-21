#pragma once

#include <type_traits>
#include <tuple>

struct coo_array_tag {};
struct csr_array_tag {};
struct csc_array_tag {};

template <typename Array, typename = std::void_t<>>
struct is_coo_array : std::false_type {};

template <typename Array, typename = std::void_t<>>
struct is_csr_array : std::false_type {};

template <typename Array, typename = std::void_t<>>
struct is_csc_array : std::false_type {};

template <typename Array>
struct is_coo_array<Array, std::void_t<typename Array::tag>>
    : std::is_same<typename Array::tag, coo_array_tag> {};

template <typename Array>
struct is_csr_array<Array, std::void_t<typename Array::tag>>
    : std::is_same<typename Array::tag, csr_array_tag> {};

template <typename Array>
struct is_csc_array<Array, std::void_t<typename Array::tag>>
    : std::is_same<typename Array::tag, csc_array_tag> {};

template <typename Array>
inline constexpr bool is_coo_array_v = is_coo_array<std::decay_t<Array>>::value;

template <typename Array>
inline constexpr bool is_csr_array_v = is_csr_array<std::decay_t<Array>>::value;

template <typename Array>
inline constexpr bool is_csc_array_v = is_csc_array<std::decay_t<Array>>::value;

template <std::size_t Axis, typename Array, typename = std::void_t<>>
struct coord_type;

template <std::size_t Axis, typename Array>
struct coord_type<Axis, Array, std::enable_if_t<is_coo_array_v<Array>>> {
  using type = std::tuple_element_t<Axis, typename Array::coord_type>;
};

template <std::size_t Axis, typename Array>
using coord_type_t = typename coord_type<Axis, std::decay_t<Array>>::type;