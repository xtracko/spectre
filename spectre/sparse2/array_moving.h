#pragma once

#include "array_canonical.h"
#include "array_utility.h"
#include "array_view.h"
#include <vector>
/*
template <unsigned Axis, typename Array>
auto moving_min(Array&& array, const coord_type_t<Axis, Array> window)
    -> std::enable_if_t<is_coo_array_v<Array>, ResultArray> {
  if (array.empty())
    return ResultArray::empty(array.shape());

  static_assert(Axis < 2);

  if constexpr (Axis == 0) {
    to_canonical(array, row_major_order{});
  } else {
    to_canonical(array, col_major_order{});
  }

  const auto coords = array.coords().first(array.size() - 1);

  auto expansions = std::vector<coord_type_t<Axis, Array>>{array.size()};
  auto expansions_sum = typename Array::index_type{0};

#pragma omp parallel for simd reduction(+ : expansions_sum) schedule(static,
512) for (auto iter = std::cbegin(coords), out = std::begin(expansions_sum);
       iter < std::cend(coords); ++iter, ++out) {
    auto&& coords_a = *iter;
    auto&& coords_b = *(iter + 1);

    auto&& axis_a = std::get<Axis>(coords_a);
    auto&& axis_b = std::get<Axis>(coords_b);

    auto distance = axis_b - axis_a;

    coord_type_t<Axis, Array> expansion =
        (extract_except<Axis>(coords_a) == extract_except<Axis>(coords_b))
            ? (distance < 2 * window) ? (distance) : (2 * window - 1)
            : (((axis_a < window) ? (axis_a + 1) : window) +
               ((axis_b > shape<Axis>(array) - window)
                    ? (shape<Axis>(array) - axis_b - 1)
                    : (window - 1)));
    expansions_sum += expansion;
    *out = expansion;
  }

  return ResultArray::empty(array.shape());
}
*/

template <
    unsigned Axis,
    typename Array,
    typename = std::enable_if_t<is_coo_array_v<Array> && is_matrix_v<Array>>>
decltype(auto) moving_min(Array&& array) {
  using array_type = std::decay_t<Array>;

  if (array.empty())
    return array_type::empty(array.shape());

  if (!is_canonical<Axis>(array))
    to_canonical<Axis>(array);

  return array_type::emoty(array.shape());
}