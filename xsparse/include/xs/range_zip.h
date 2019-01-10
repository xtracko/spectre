#pragma once

#include <tuple>
#include <xs/range_traits.h>

namespace xs::range {
  template <typename... Ts> struct proxy_zip : std::tuple<Ts...> {
    using std::tuple<Ts...>::tuple;
    using std::tuple<Ts...>::swap;
    using std::tuple<Ts...>::operator=;
  };

  template <typename... Ts>
  void swap(proxy_zip<Ts...> &&a, proxy_zip<Ts...> &&b) {
    a.swap(b);
  }

  template <typename... Bases> struct iter_zip {
    using value_type = proxy_zip<iter_value_t<Bases>...>;
    using pointer = proxy_zip<iter_pointer_t<Bases>...>;
    using reference = proxy_zip<iter_reference_t<Bases>...>;
    using difference_type = std::common_type_t<iter_difference_t<Bases>...>;
    using iterator_category = std::common_type_t<iter_category_t<Bases>...>;

    template <typename = enable_forward_iterator_t<iter_zip, void>>
    constexpr explicit iter_zip()
        : _bases{} {}

    template <typename = enable_input_iterator_t<iter_zip, void>>
    constexpr explicit iter_zip(Bases... bases)
        : _bases{std::move(bases)...} {}

    constexpr enable_input_iterator_t<iter_zip, bool>
    operator==(const iter_zip &other) const {
      return std::get<0>(_bases) == std::get<0>(other._bases);
    }

    constexpr enable_input_iterator_t<iter_zip, bool>
    operator!=(const iter_zip &other) const {
      return std::get<0>(_bases) != std::get<0>(other._bases);
    }

    constexpr enable_random_access_iterator_t<iter_zip, bool>
    operator<(const iter_zip &other) const {
      return std::get<0>(_bases) < std::get<0>(other._bases);
    }

    constexpr enable_random_access_iterator_t<iter_zip, bool>
    operator>(const iter_zip &other) const {
      return std::get<0>(_bases) > std::get<0>(other._bases);
    }

    constexpr enable_random_access_iterator_t<iter_zip, bool>
    operator<=(const iter_zip &other) const {
      return std::get<0>(_bases) <= std::get<0>(other._bases);
    }

    constexpr enable_random_access_iterator_t<iter_zip, bool>
    operator>=(const iter_zip &other) const {
      return std::get<0>(_bases) >= std::get<0>(other._bases);
    }

    constexpr enable_input_iterator_t<iter_zip, iter_zip> &operator++() {
      return std::apply([](auto &... bases) { (++bases, ...); }, _bases), *this;
    }

    constexpr enable_bidirectional_iterator_t<iter_zip, iter_zip> &
    operator--() {
      return std::apply([](auto &... bases) { (--bases, ...); }, _bases), *this;
    }

    constexpr const enable_input_iterator_t<iter_zip, iter_zip>
    operator++(int) {
      auto inc = [](auto &... bases) -> iter_zip { return {bases++...}; };
      return std::apply(std::move(inc), _bases);
    }

    constexpr const enable_bidirectional_iterator_t<iter_zip, iter_zip>
    operator--(int) {
      auto dec = [](auto &... bases) -> iter_zip { return {bases--...}; };
      return std::apply(std::move(dec), _bases);
    }

    constexpr enable_random_access_iterator_t<iter_zip, difference_type>
    operator-(const iter_zip &other) const {
      return std::get<0>(_bases) - std::get<0>(other._bases);
    }

    constexpr enable_random_access_iterator_t<iter_zip, iter_zip>
    operator+(const difference_type n) const {
      auto add = [n](auto &... bases) -> iter_zip { return {(bases + n)...}; };
      return std::apply(std::move(add), _bases);
    }

    constexpr enable_random_access_iterator_t<iter_zip, iter_zip>
    operator-(const difference_type n) const {
      auto sub = [n](auto &... bases) -> iter_zip { return {(bases - n)...}; };
      return std::apply(std::move(sub), _bases);
    }

    constexpr enable_random_access_iterator_t<iter_zip, iter_zip> &
    operator+=(const difference_type n) {
      auto add = [n](auto &... bases) -> void { ((bases += n), ...); };
      return std::apply(std::move(add), _bases), *this;
    }

    constexpr enable_random_access_iterator_t<iter_zip, iter_zip> &
    operator-=(const difference_type n) {
      auto sub = [n](auto &... bases) -> void { ((bases -= n), ...); };
      return std::apply(std::move(sub), _bases), *this;
    }

    constexpr enable_random_access_iterator_t<iter_zip, reference>
    operator[](const difference_type n) const {
      auto deref = [n](auto &... bases) -> reference { return {bases[n]...}; };
      return std::apply(std::move(deref), _bases);
    }

    constexpr enable_input_iterator_t<iter_zip, reference> operator*() const {
      auto deref = [](auto &... bases) -> reference { return {*bases...}; };
      return std::apply(std::move(deref), _bases);
    }

  private:
    std::tuple<Bases...> _bases;
  };

  template <typename... Bases> struct zip {
    using iterator = iter_zip<typename Bases::iterator...>;
    using const_iterator = iter_zip<typename Bases::const_iterator...>;

    template <typename... Ranges>
    constexpr explicit zip(Ranges &&... ranges)
        : _bases{std::forward<Ranges>(ranges)...} {}

    constexpr zip(const zip &) = default;
    constexpr zip(zip &&) noexcept = default;

    constexpr zip &operator=(const zip &) = default;
    constexpr zip &operator=(zip &&) noexcept = default;

    constexpr void swap(zip &) noexcept = default;

    constexpr const_iterator begin() const {
      return std::apply(
          [](auto &... bases) -> const_iterator {
            using std::begin;
            return {begin(bases)...};
          },
          _bases);
    }

    constexpr const_iterator end() const {
      return std::apply(
          [](auto &... bases) -> const_iterator {
            using std::end;
            return {end(bases)...};
          },
          _bases);
    }

  private:
    std::tuple<Bases...> _bases;
  };

  template <typename... Ranges> zip(Ranges &&...)->zip<Ranges...>;

  template <typename... Bases>
  void swap(zip<Bases...> &lhs, zip<Bases...> &rhs) noexcept {
    lhs.swap(rhs);
  }
} // namespace xs::range