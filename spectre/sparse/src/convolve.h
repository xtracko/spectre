#pragma once

#include "ranges.h"
#include "span.h"

namespace spectre {
  template <typename I, typename J, typename D>
  void convolve_csr_dv(cspan<I> const A_rows, cspan<I> const A_cols,
                       cspan<D> const A_data, cspan<D> const coeffs,
                       span<J> B_cols, span<D> B_data, I const n_cols) noexcept
  {
    auto const window = static_cast<J>(coeffs.size());
    auto const wnd_lhs = window / 2;
    auto const wnd_rhs = (window + 1) / 2;

    auto out_col = B_cols.begin();
    auto out_val = B_data.begin();

    for (auto const [a, b] : adjacent(A_rows)) {
      auto const cols = A_cols.slice(a, b);
      auto const data = A_data.slice(a, b);

      auto start = -wnd_lhs;
      auto stop = n_cols - wnd_rhs;
      auto col_iter = cols.begin();
      auto val_iter = data.begin();

      while (start < stop && col_iter < cols.end()) {
        start = std::max(start, *col_iter - window + 1);

        auto value = static_cast<D>(0);
        auto krn_col_iter = col_iter;
        auto krn_val_iter = val_iter;
        auto coeff_iter = std::make_reverse_iterator(coeffs.end());

        for (auto index = start; index < start + window;
             ++index, ++coeff_iter) {
          if (krn_col_iter < cols.end() && index == *krn_col_iter) {
            value += (*coeff_iter) * (*krn_val_iter);
            ++krn_col_iter;
            ++krn_val_iter;
          }
        }

        assert(out_col < B_cols.end());
        assert(out_val < B_data.end());

        *out_col++ = start + wnd_lhs;
        *out_val++ = value;

        if (++start > *col_iter) {
          ++col_iter;
          ++val_iter;
        }
      }
    }
  }
} // namespace spectre