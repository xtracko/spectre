#pragma once

#include <iterator>
#include <tuple>

namespace range {
  template <typename... Ts> struct proxy_zip : std::tuple<Ts...> {
    using std::tuple<Ts...>::tuple;
    using std::tuple<Ts...>::swap;
    using std::tuple<Ts...>::operator=;
  };

  template <typename... Bs> struct iter_zip {
    using iterator_category = std::common_type_t<
        typename std::iterator_traits<Bs>::iterator_category...>;
    using difference_type = std::common_type_t<
        typename std::iterator_traits<Bs>::difference_type...>;
    using value_type =
        proxy_zip<typename std::iterator_traits<Bs>::value_type...>;
    using pointer = proxy_zip<typename std::iterator_traits<Bs>::pointer...>;
    using reference =
        proxy_zip<typename std::iterator_traits<Bs>::reference...>;

    constexpr iter_zip() = default;

    constexpr explicit iter_zip(Bs... bases)
        : _bases{std::move(bases)...} {
    }

    constexpr reference operator*() const {
      return std::apply([](auto... bs) { return reference{*bs...}; }, _bases);
    }

    constexpr reference operator[](const difference_type n) const {
      return *(*this + n);
    }

    constexpr iter_zip& operator++() {
      std::apply([](auto... bs) { (++bs, ...); }, _bases);
      return *this;
    }

    constexpr iter_zip& operator--() {
      std::apply([](auto... bs) { (--bs, ...); }, _bases);
      return *this;
    }

    constexpr const iter_zip operator++(int) {
      auto self = *this;
      ++(*this);
      return self;
    }

    constexpr const iter_zip operator--(int) {
      auto self = *this;
      --(*this);
      return self;
    }

    constexpr difference_type operator-(const iter_zip& other) const {
      return std::get<0>(_bases) - std::get<0>(other._bases);
    }

    constexpr bool operator==(const iter_zip& other) const {
      return std::get<0>(_bases) == std::get<0>(other._bases);
    }

    constexpr bool operator!=(const iter_zip& other) const {
      return !(*this == other);
    }

    constexpr bool operator<(const iter_zip& other) const {
      return *this - other < 0;
    }

    constexpr bool operator>(const iter_zip& other) const {
      return *this - other > 0;
    }

    constexpr bool operator<=(const iter_zip& other) const {
      return *this - other <= 0;
    }

    constexpr bool operator>=(const iter_zip& other) const {
      return *this - other >= 0;
    }

    constexpr iter_zip operator+(const difference_type n) const {
      auto self = *this;
      return self += n;
    }

    constexpr iter_zip operator-(const difference_type n) const {
      auto self = *this;
      return self -= n;
    }

    constexpr iter_zip& operator+=(const difference_type n) {
      std::apply([=](auto... bs) { ((bs += n), ...); }, _bases);
      return *this;
    }

    constexpr iter_zip& operator-=(const difference_type n) {
      return *this += -n;
    }

  private:
    std::tuple<Bs...> _bases;
  };

  template <typename... Bs> struct zip {
    using index_type = std::common_type_t<typename Bs::index_type...>;
    using iterator = iter_zip<typename Bs::iterator...>;
    using const_iterator = iter_zip<typename Bs::const_iterator...>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using reverse_const_iterator = std::reverse_iterator<const_iterator>;

    constexpr zip() = default;
    constexpr zip& operator=(const zip&) = default;

    constexpr explicit zip(Bs... bs)
        : _bases{std::move(bs)...} {
    }

    constexpr bool empty() const {
      return std::empty(std::get<0>(_bases));
    }

    constexpr index_type size() const {
      return std::size(std::get<0>(_bases));
    }

    /****************************************************************************/

    constexpr iterator begin() const {
      return std::apply([](auto&... bs) { return iterator{bs.begin()...}; },
                        _bases);
    }

    constexpr const_iterator cbegin() const {
      return std::apply(
          [](auto&&... bs) { return const_iterator{bs.cbegin()...}; }, _bases);
    }

    constexpr iterator end() const {
      return std::apply([](auto&&... bs) { return iterator{bs.end()...}; },
                        _bases);
    }

    constexpr const_iterator cend() const {
      return std::apply(
          [](auto&&... bs) { return const_iterator{bs.cend()...}; }, _bases);
    }

    /****************************************************************************/

    constexpr reverse_iterator rbegin() const {
      return std::apply(
          [](auto&... bs) { return reverse_iterator{bs.rbegin()...}; }, _bases);
    }

    constexpr reverse_const_iterator crbegin() const {
      return std::apply(
          [](auto&&... bs) { return reverse_const_iterator{bs.crbegin()...}; },
          _bases);
    }

    constexpr reverse_iterator rend() const {
      return std::apply(
          [](auto&&... bs) { return reverse_iterator{bs.rend()...}; }, _bases);
    }

    constexpr reverse_const_iterator crend() const {
      return std::apply(
          [](auto&&... bs) { return reverse_const_iterator{bs.crend()...}; },
          _bases);
    }

    /****************************************************************************/

    constexpr auto first(const index_type count) const {
      return std::apply([=](auto&&... bs) { return zip{bs.first(count)...}; },
                        _bases);
    }

    constexpr auto last(const index_type count) const {
      return std::apply([=](auto&&... bs) { return zip{bs.last(count)...}; },
                        _bases);
    }

    constexpr auto subspan(const index_type offset) const {
      return std::apply(
          [=](auto&&... bs) { return zip{bs.subspan(offset)...}; }, _bases);
    }

    constexpr auto subspan(const index_type offset,
                           const index_type count) const {
      return std::apply(
          [=](auto&&... bs) { return zip{bs.subspan(offset, count)...}; },
          _bases);
    };

  private:
    std::tuple<Bs...> _bases;
  };
} // namespace range

namespace std {
  template <typename... Ts>
  void swap(range::proxy_zip<Ts...>&& a, range::proxy_zip<Ts...>&& b) {
    a.swap(b);
  }

  template <std::size_t Index, typename... Ts>
  constexpr decltype(auto) get(range::proxy_zip<Ts...>& proxy) noexcept {
    return std::get<Index>(static_cast<std::tuple<Ts...>&>(proxy));
  }

  template <std::size_t Index, typename... Ts>
  constexpr decltype(auto) get(range::proxy_zip<Ts...>&& proxy) noexcept {
    return std::get<Index>(static_cast<std::tuple<Ts...>&&>(proxy));
  }

  template <std::size_t Index, typename... Ts>
  constexpr decltype(auto) get(const range::proxy_zip<Ts...>& proxy) noexcept {
    return std::get<Index>(static_cast<const std::tuple<Ts...>&>(proxy));
  }

  template <std::size_t Index, typename... Ts>
  constexpr decltype(auto) get(const range::proxy_zip<Ts...>&& proxy) noexcept {
    return std::get<Index>(static_cast<const std::tuple<Ts...>&&>(proxy));
  }
} // namespace std