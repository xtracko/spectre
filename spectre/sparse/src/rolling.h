#pragma once

#include "span.h"
#include <algorithm>
#include <vector>

namespace spectre {
  template <typename I, typename J>
  J rolling_alloc_csr(cspan<I> const A_rows, cspan<I> const A_cols,
                      span<J> const B_rows, I const A_n_cols,
                      J const window) noexcept
  {
    auto out = B_rows.begin();
    auto size = static_cast<J>(0);
    auto const wnd_lhs = window / 2;
    auto const wnd_rhs = (window + 1) / 2;

    *out++ = size;
    for (auto const [a, b] : adjacent(A_rows)) {
      auto beg = A_cols.begin() + a;
      auto end = A_cols.begin() + b;

      if (beg != end) {
        size += std::min(wnd_lhs, static_cast<J>(beg[0]));
        for (--end; beg < end; ++beg)
          size += std::min(window, static_cast<J>(beg[1] - beg[0]));
        size += std::min(wnd_rhs, static_cast<J>(A_n_cols - beg[0]));
      }
      *out++ = size;
    }
    return size;
  }

  template <typename I, typename J, typename D,
            template <typename, typename> typename Krn>
  void rolling_csr(cspan<I> const A_rows, cspan<I> const A_cols,
                   cspan<D> const A_data, span<J> const B_cols,
                   span<D> const B_data, I const A_n_cols,
                   J const window) noexcept
  {
    auto kernel = Krn<J, D>{window};
    auto const wnd_lhs = (window - 1) / 2;

    auto out_col = B_cols.begin();
    auto out_val = B_data.begin();

    for (auto const [a, b] : adjacent(A_rows)) {
      auto const cols = A_cols.slice(a, b);
      auto const data = A_data.slice(a, b);

      auto start = -wnd_lhs;
      auto stop = A_n_cols - wnd_lhs;
      auto col_iter = cols.begin();
      auto val_iter = data.begin();

      while (start < stop && col_iter < cols.end()) {
        start = std::max(start, *col_iter - window + 1);

        kernel.init();
        auto krn_col_iter = col_iter;
        auto krn_val_iter = val_iter;

        for (auto index = start; index < start + window; ++index) {
          if (krn_col_iter < cols.end() && index == *krn_col_iter) {
            kernel.push(*krn_val_iter);
            ++krn_col_iter;
            ++krn_val_iter;
          } else {
            kernel.push(0);
          }
        }

        assert(out_col < B_cols.end());
        assert(out_val < B_data.end());
        assert(0 <= start + wnd_lhs);
        assert(start + wnd_lhs < A_n_cols);

        *out_col++ = start + wnd_lhs;
        *out_val++ = kernel.pop();

        if (++start > *col_iter) {
          ++col_iter;
          ++val_iter;
        }
      }
    }
  }

  template <typename I, typename T> struct min_kernel {
    explicit min_kernel(I const) noexcept
    {}

    void init() noexcept
    {
      _result = std::numeric_limits<T>::max();
    }

    void push(T const value) noexcept
    {
      _result = std::min(_result, value);
    }

    auto pop() const noexcept
    {
      return _result;
    }

  private:
    T _result;
  };

  template <typename I, typename T> struct max_kernel {
    explicit max_kernel(I const) noexcept
    {}

    void init() noexcept
    {
      _result = std::numeric_limits<T>::min();
    }

    void push(T const value) noexcept
    {
      _result = std::max(_result, value);
    }

    auto pop() const noexcept
    {
      return _result;
    }

  private:
    T _result;
  };

  template <typename I, typename T> struct mean_kernel {
    explicit mean_kernel(I const window) noexcept : _window{window}
    {}

    void init() noexcept
    {
      _result = 0;
    }

    void push(T const value) noexcept
    {
      _result += value;
    }

    auto pop() const noexcept
    {
      return _result / _window;
    }

  private:
    T _result;
    I const _window;
  };

  template <typename I, typename T> struct median_kernel {
    explicit median_kernel(I const window) noexcept : _buffer(window)
    {}

    void init() noexcept
    {
      _buffer.clear();
    }

    void push(T const value) noexcept
    {
      _buffer.push_back(value);
    }

    auto pop() noexcept
    {
      std::sort(_buffer.begin(), _buffer.end());
      return _buffer[_buffer.size() / 2];
    }

  private:
    std::vector<T> _buffer;
  };
} // namespace spectre
