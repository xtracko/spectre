#pragma once

#include "ranges.h"
#include "span.h"
#include <cmath>
#include <numeric>

namespace spectre {
  template <typename I, typename D>
  void stdev_csr(cspan<I> const rows, cspan<D> const data, I const n_cols,
                 span<D> const result) noexcept
  {
    auto out = result.begin();

    for (auto const [a, b] : adjacent(rows)) {
      auto sum = static_cast<D>(0);
      auto sum2 = static_cast<D>(0);

      for (auto const value : data.slice(a, b)) {
        sum += value;
        sum2 += value * value;
      }

      auto const mean = sum / n_cols;
      auto const var = sum2 / n_cols - (mean * mean);
      *out++ = std::sqrt(var);
    }
  }
} // namespace spectre
