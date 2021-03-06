#pragma once

#include "ranges.h"
#include "span.h"

namespace spectre {
  template <typename I>
  bool is_canonical_coo(cspan<I> const A_rows, cspan<I> const A_cols) noexcept
  {
    bool sorted = true;
    for (auto const [i, j] : zip(adjacent(A_rows), adjacent(A_cols)))
      sorted &= (std::get<0>(i) < std::get<1>(i)) |
                ((std::get<0>(i) == std::get<1>(i)) &
                 (std::get<0>(j) < std::get<1>(j)));
    return sorted;
  }

  template <typename I>
  bool is_canonical_csr(cspan<I> const A_rows, cspan<I> const A_cols) noexcept
  {
    bool sorted = true;
    for (auto const [a, b] : adjacent(A_rows))
      for (auto const [u, v] : adjacent(A_cols.slice(a, b)))
        sorted &= u < v;
    return sorted;
  }

  template <typename I>
  auto canonical_alloc_csr(cspan<I> const A_rows, cspan<I> const A_cols,
                           span<I> const B_rows) noexcept
  {
    auto size = static_cast<I>(0);
    auto iter = B_rows.begin();

    *iter++ = size;
    for (auto const [a, b] : adjacent(A_rows)) {
      for (auto const [u, v] : adjacent(A_cols.slice(a, b)))
        size += u != v;
      *iter++ = size;
    }
    return size;
  }
} // namespace spectre
