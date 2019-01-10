#pragma once

#include "range_view.h"
#include <array>
#include <xs/range_traits.h>

namespace xs::range {
  template <typename Base> struct iter_adjacent {
    using pointer = std::array<iter_pointer_t<Base>, 2>;
    using reference = std::array<iter_reference_t<Base>, 2>;
    using value_type = std::array<iter_value_t<Base>, 2>;
    using difference_type = iter_difference_t<Base>;
    using iterator_category = iter_category_t<Base>;

    constexpr enable_input_iterator_t<Base, iter_adjacent> &operator++() {
      return ++_base, *this;
    }

    constexpr enable_bidirectional_iterator_t<Base, iter_adjacent> &
    operator--() {
      return --_base, *this;
    }

    constexpr const enable_input_iterator_t<Base, iter_adjacent>
    operator++(int) {
      return {_base++};
    }

    constexpr const enable_bidirectional_iterator_t<Base, iter_adjacent>
    operator--(int) {
      return {_base--};
    }

    constexpr enable_input_iterator_t<Base, bool>
    operator==(const iter_adjacent &other) const {
      return _base == other._base;
    }

    constexpr enable_input_iterator_t<Base, bool>
    operator!=(const iter_adjacent &other) const {
      return _base != other._base;
    }

    constexpr enable_random_access_iterator_t<Base, bool>
    operator<(const iter_adjacent &other) const {
      return _base < other._base;
    }

    constexpr enable_random_access_iterator_t<Base, bool>
    operator>(const iter_adjacent &other) const {
      return _base > other._base;
    }

    constexpr enable_random_access_iterator_t<Base, bool>
    operator<=(const iter_adjacent &other) const {
      return _base <= other._base;
    }

    constexpr enable_random_access_iterator_t<Base, bool>
    operator>=(const iter_adjacent &other) const {
      return _base >= other._base;
    }

    constexpr enable_random_access_iterator_t<Base, difference_type>
    operator-(const iter_adjacent &other) const {
      return _base - other._base;
    }

    constexpr enable_random_access_iterator_t<Base, iter_adjacent>
    operator+(const difference_type n) const {
      return {_base + n};
    }

    constexpr enable_random_access_iterator_t<Base, iter_adjacent>
    operator-(const difference_type n) const {
      return {_base - n};
    }

    constexpr enable_random_access_iterator_t<Base, iter_adjacent> &
    operator+=(const difference_type n) {
      return _base += n, *this;
    }

    constexpr enable_random_access_iterator_t<Base, iter_adjacent> &
    operator-=(const difference_type n) {
      return _base -= n, *this;
    }

    constexpr enable_random_access_iterator_t<Base, reference>
    operator[](const difference_type n) const {
      return {_base[n], _base[n + 1]};
    }

    constexpr enable_input_iterator_t<Base, reference> operator*() const {
      return {*_base, *std::next(_base)};
    }

  private:
    Base _base;
  };
} // namespace xs::range

/*
template <typename Iter> void foo(Iter beg, Iter end) {
  if (beg == end)
    return;

  for (--end; beg != end; ++beg)
    auto item = {beg[0], beg[1]};
}
*/