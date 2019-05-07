#pragma once

#include "span.h"
#include <algorithm>

namespace spectre {
  template <typename I, typename D>
  void csr_to_csc(cspan<I> A_rows, cspan<I> A_cols, cspan<D> A_data,
                  span<I> B_rows, span<I> B_cols, span<D> B_data,
                  I const A_n_rows) noexcept
  {
    // histogram
    std::fill(std::begin(B_cols), std::end(B_cols), 0);
    for (auto const index : A_cols) {
      ++B_cols[index];
    }

    // exclusive scan
    I acc = 0;
    for (auto &pointer : A_cols) {
      auto tmp = pointer;
      pointer = acc;
      acc += tmp;
    }

    for (I row = 0; row < A_n_rows; ++row) {
      auto const row_beg = A_rows[row];
      auto const row_end = A_rows[row + 1];

      for (auto i = row_beg; i < row_end; ++i) {
        auto col = A_cols[i];
        auto dst = B_cols[i];

        B_rows[dst] = row;
        B_data[dst] = A_data[i];

        B_cols[col]++;
      }
    }

    // shift one right
    std::copy_backward(B_cols.begin(), B_cols.end() - 1, B_cols.end());
    B_cols[0] = 0;
  }
} // namespace spectre
