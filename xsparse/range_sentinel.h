#pragma once

#include "range_traits.h"

namespace xsparse::range {
  template <typename Base> struct sentinel {
    using difference_type = difference_type_t<Base>;
    using iterator_category = iterator_category_t<Base>;

    constexpr explicit sentinel(Base base)
        : _base(std::move(base)) {}

    template <typename = std::enable_if_t<is_forward_iterator_v<Base>>>
    constexpr explicit sentinel()
        : _base{} {}

    constexpr std::enable_if_t<is_input_iterator_v<Base>, bool>
    operator==(const sentinel &other) const {
      return _base == other._base;
    }

    constexpr std::enable_if_t<is_input_iterator_v<Base>, bool>
    operator!=(const sentinel &other) const {
      return _base != other._base;
    }

    constexpr std::enable_if_t<is_input_iterator_v<Base>, sentinel &>
    operator++() {
      ++_base;
      return *this;
    }

    constexpr std::enable_if_t<is_bidirectional_iterator_v<Base>, sentinel &>
    operator--() {
      --_base;
      return *this;
    }

    constexpr std::enable_if_t<is_input_iterator_v<Base>, sentinel>
    operator++(int) {
      return sentinel{_base++};
    }

    constexpr std::enable_if_t<is_bidirectional_iterator_v<Base>, sentinel>
    operator--(int) {
      return sentinel{_base--};
    }

    constexpr std::enable_if_t<is_random_access_iterator_v<Base>, sentinel>
    operator+(const difference_type count) const {
      return sentinel{_base + count};
    }

    constexpr std::enable_if_t<is_random_access_iterator_v<Base>, sentinel>
    operator-(const difference_type count) const {
      return sentinel{_base - count};
    }

    constexpr std::enable_if_t<is_random_access_iterator_v<Base>, sentinel &>
    operator+=(const difference_type count) {
      _base += count;
      return *this;
    }

    constexpr std::enable_if_t<is_random_access_iterator_v<Base>, sentinel &>
    operator-=(const difference_type count) {
      _base -= count;
      return *this;
    }

    constexpr std::enable_if_t<is_random_access_iterator_v<Base>, bool>
    operator<(const sentinel &other) const {
      return _base < other._base;
    }

    constexpr std::enable_if_t<is_random_access_iterator_v<Base>, bool>
    operator>(const sentinel &other) const {
      return _base > other._base;
    }

    constexpr std::enable_if_t<is_random_access_iterator_v<Base>, bool>
    operator<=(const sentinel &other) const {
      return _base <= other._base;
    }

    constexpr std::enable_if_t<is_random_access_iterator_v<Base>, bool>
    operator>=(const sentinel &other) const {
      return _base >= other._base;
    }

    constexpr std::enable_if_t<is_random_access_iterator_v<Base>,
                               difference_type>
    operator-(const sentinel &other) const {
      return _base - other._base;
    }

  private:
    Base _base;
  };
} // namespace xsparse::range