#pragma once

#include "span.h"

namespace spectre {
  template <typename I, typename D>
  void maxclip_csr_spmat_plus_dvec_nonnegative(
      cspan<I> const A_rows, cspan<I> const A_cols, span<D> const A_data,
      cspan<I> const B_rows, cspan<I> const B_cols, cspan<D> const B_data,
      cspan<D> const C_data)
  {
    bool any_negative = false;
    for (auto const value : B_data)
      any_negative |= value < 0;
    for (auto const value : C_data)
      any_negative |= value < 0;
    if (any_negative)
      throw std::domain_error{"max-clip algorithm can only handle sparse "
                              "matrices with all values being non-negative"};

    auto const i_end = static_cast<std::ptrdiff_t>(A_rows.size()) - 1;
    for (std::ptrdiff_t i = 0; i < i_end; ++i) {
      auto const C_val = C_data[i];
      auto A_beg = A_rows[i];
      auto B_beg = B_rows[i];
      auto const A_end = A_rows[i + 1];
      auto const B_end = B_rows[i + 1];

      for (; A_beg < A_end; ++A_beg) {
        auto const A_col = A_cols[A_beg];
        auto const A_val = A_data[A_beg];

        while (B_beg < B_end && B_cols[B_beg] < A_col)
          ++B_beg;

        bool const aligned = B_beg < B_end && B_cols[B_beg] == A_col;
        auto const max_val = aligned ? (B_data[B_beg] + C_val) : C_val;

        A_data[A_beg] = (A_val > max_val) ? max_val : A_val;
      }
    }
  }
}