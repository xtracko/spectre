#pragma once

#include <xs/range_traits.h>

namespace xs::range {
  template <typename Iter> struct view {
    using iterator = Iter;

    constexpr view(view const &) = default;
    constexpr view(view &&) noexcept = default;

    constexpr view &operator=(const view &) = default;
    constexpr view &operator=(view &&) noexcept = default;

    constexpr view(iterator beg, iterator end)
        : _beg{std::move(beg)}
        , _end{std::move(end)} {}

    constexpr bool empty() const {
      return _beg == _end;
    }

    constexpr auto size() const {
      return std::distance(_beg, _end);
    }

    constexpr iterator begin() const {
      return _beg;
    }

    constexpr iterator end() const {
      return _end;
    }

  private:
    iterator _beg;
    iterator _end;
  };
} // namespace xs::range
