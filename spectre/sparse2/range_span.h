#pragma once

#include <iterator>
#include <type_traits>

namespace range {
  template <typename T> struct span {
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using index_type = std::ptrdiff_t;
    using pointer = std::add_pointer_t<T>;
    using reference = std::add_lvalue_reference_t<T>;
    using iterator = std::add_pointer_t<T>;
    using const_iterator = std::add_pointer_t<std::add_const_t<T>>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using reverse_const_iterator = std::reverse_iterator<const_iterator>;

    constexpr span() noexcept
        : span(nullptr, nullptr) {
    }

    constexpr span(const span&) noexcept = default;
    constexpr span& operator=(const span&) noexcept = default;

    constexpr span(pointer beg, pointer end)
        : _beg(beg)
        , _end(end) {
    }

    constexpr span(pointer ptr, index_type count)
        : span(ptr, ptr + count) {
    }

    constexpr bool empty() const {
      return _beg == _end;
    }

    constexpr index_type size() const {
      return std::distance(_beg, _end);
    }

    constexpr iterator begin() const {
      return iterator{_beg};
    }

    constexpr const_iterator cbegin() const {
      return const_iterator{_beg};
    }

    constexpr iterator end() const {
      return iterator{_end};
    }

    constexpr const_iterator cend() const {
      return const_iterator{_end};
    }

    constexpr reverse_iterator rbegin() const {
      return reverse_iterator{_beg};
    }

    constexpr reverse_const_iterator crbegin() const {
      return reverse_const_iterator{_beg};
    }

    constexpr reverse_iterator rend() const {
      return reverse_iterator{_end};
    }

    constexpr reverse_const_iterator crend() const {
      return reverse_const_iterator{_end};
    }

    constexpr span first(const index_type count) const {
      return span{_beg, count};
    };

    constexpr span last(const index_type count) const {
      return span{_end - count, _end};
    };

    constexpr span subspan(const index_type offset) const {
      return span{_beg + offset, _end};
    };

    constexpr span subspan(const index_type offset,
                           const index_type count) const {
        return span{_beg + offset, count};
    };

  private:
    pointer _beg;
    pointer _end;
  };
} // namespace range