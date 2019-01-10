#pragma once

#include <type_traits>
#include <utility>

namespace xs {
  template <typename Array, typename Shape>
  std::enable_if_t<std::is_constructible_v<Array, Shape>, Array>
  empty(Shape &&shape) {
    return Array{std::forward<Shape>(shape)};
  }

  template <typename Array, typename Shape, typename Size>
  std::enable_if_t<std::is_constructible_v<Array, Shape, Size>, Array>
  zeros(Shape &&shape, Size &&size) {
    return Array{std::forward<Shape>(shape), std::forward<Size>(size)};
  }

  template <typename Array> constexpr auto empty_like(Array &&array) {
    return empty<std::decay_t<Array>>(array.shape());
  }

  template <typename Array> constexpr auto zeros_like(Array &&array) {
    return zeros<std::decay_t<Array>>(array.shape(), array.size());
  }
} // namespace xs
