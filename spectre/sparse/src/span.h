#pragma once

#include <cassert>
#include <type_traits>

namespace spectre {
  template <typename T> struct span {
    using pointer = T *;
    using iterator = T *;
    using reference = T &;
    using value_type = std::remove_cv_t<T>;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;

    constexpr span() noexcept = default;
    constexpr span(span &&) noexcept = default;
    constexpr span(span const &) noexcept = default;
    constexpr span &operator=(span &&) noexcept = default;
    constexpr span &operator=(span const &) noexcept = default;

    constexpr explicit span(pointer beg, pointer end) noexcept
        : _beg{beg}, _end{end}
    {
      assert(_beg <= _end);
    }

    constexpr explicit span(pointer p, size_type n) noexcept
        : _beg{p}, _end{p + n}
    {
      assert(_beg <= _end);
    }

    constexpr bool empty() const noexcept
    {
      return _beg == _end;
    }

    constexpr size_type size() const noexcept
    {
      return static_cast<size_type>(_end - _beg);
    }

    constexpr iterator begin() const noexcept
    {
      return _beg;
    }

    constexpr iterator end() const noexcept
    {
      return _end;
    }

    constexpr span slice(size_type const a, size_type const b) const noexcept
    {
      assert((_beg + a <= _end) && (_beg + b <= _end));
      return span{_beg + a, _beg + b};
    }

    constexpr reference operator[](size_type const n) const noexcept
    {
      assert(_beg + n < _end);
      return _beg[n];
    }

    constexpr reference front() const noexcept
    {
      return operator[](0);
    }

    constexpr reference back() const noexcept
    {
      return operator[](size() - 1);
    }

  private:
    pointer _beg = nullptr;
    pointer _end = nullptr;
  };

  template <typename T> using cspan = span<T const>;
} // namespace spectre
