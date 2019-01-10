#pragma once

#include "range_traits.h"
#include "range_zip.h"
#include <stdexcept>
#include <xs/array_traits.h>
#include <xs/array_utils.h>

namespace xs {
  template <typename Array, typename Kernel>
  auto stencil(Array &&array, Kernel &&kernel) {
    if (array.dims() != kernel.dims())
      throw std::invalid_argument{
          "The dimension of a kernel must match a dimension of an input array"};

    if (!is_canonical(array))
      throw std::invalid_argument{
          "An input array of a stencil operation must be in canonical order."};

    if (array.empty())
      return empty_like(std::forward<Array>(array));
  }
} // namespace xs
