#include <range_traits.h>
#include <tuple>

namespace xsparse::range {
  template <typename... Ts> struct zip_proxy : std::tuple<Ts...> {
    using std::tuple<Ts...>::tuple;
    using std::tuple<Ts...>::swap;
    using std::tuple<Ts...>::operator=;
  };

  template <typename... Bs> struct zipper {
    using difference_type = std::common_type_t<iterator_difference_t<Bs>...>;
    using iterator_category = std::common_type_t<iterator_category_t<Bs>...>;
    using value_type = zip_proxy<iterator_value_t<Bs>...>;
    using pointer = zip_proxy<iterator_pointer_t<Bs>...>;
    using reference = zip_proxy<iterator_reference_t<Bs>...>;

    constexpr explicit zipper(Bs... bases)
        : _bases{std::move(bases)...} {}

    template <typename = std::enable_if_t<is_forward_iterator_v<zipper>>>
    constexpr explicit zipper()
        : _bases{} {}

    constexpr std::enable_if_t<is_input_iterator_v<zipper>, reference>
    operator*() const {
      return std::apply([](auto &&... bs) { return reference{*bs...}; },
                        _bases);
    }

    constexpr std::enable_if_t<is_input_iterator_v<zipper>, zipper &>
    operator++() {
      std::apply([](auto &&... bs) { (++bs, ...); }, _bases);
      return *this;
    }

    constexpr std::enable_if_t<is_bidirectional_iterator_v<zipper>, zipper &>
    operator--() {
      std::apply([](auto &&... bs) { (--bs, ...); }, _bases);
      return *this;
    }

    constexpr std::enable_if_t<is_input_iterator_v<zipper>, zipper>
    operator++(int) {
      auto self = *this;
      ++(*this);
      return self;
    }

    constexpr std::enable_if_t<is_bidirectional_iterator_v<zipper>, zipper>
    operator--(int) {
      auto self = *this;
      --(*this);
      return self;
    }

    constexpr std::enable_if_t<is_input_iterator_v<zipper>, bool>
    operator==(const zipper &other) const {
      return std::get<0>(_bases) == std::get<0>(other._bases);
    }

    constexpr std::enable_if_t<is_input_iterator_v<zipper>, bool>
    operator!=(const zipper &other) const {
      return std::get<0>(_bases) != std::get<0>(other._bases);
    }

    constexpr std::enable_if_t<is_random_access_iterator_v<zipper>, zipper>
    operator+(const difference_type n) const {
      return zipper{*this} += n;
    }

    constexpr std::enable_if_t<is_random_access_iterator_v<zipper>, zipper>
    operator-(const difference_type n) const {
      return zipper{*this} -= n;
    }

    constexpr std::enable_if_t<is_random_access_iterator_v<zipper>, zipper &>
    operator+=(const difference_type n) {
      std::apply([=](auto &&... bs) { ((bs += n), ...); }, _bases);
      return *this;
    }

    constexpr std::enable_if_t<is_random_access_iterator_v<zipper>, zipper &>
    operator-=(const difference_type n) {
      return *this += -n;
    }

    constexpr std::enable_if_t<is_random_access_iterator_v<zipper>, bool>
    operator<(const zipper &other) const {
      return std::get<0>(_bases) < std::get<0>(other._bases);
    }

    constexpr std::enable_if_t<is_random_access_iterator_v<zipper>, bool>
    operator>(const zipper &other) const {
      return std::get<0>(_bases) > std::get<0>(other._bases);
    }

    constexpr std::enable_if_t<is_random_access_iterator_v<zipper>, bool>
    operator<=(const zipper &other) const {
      return std::get<0>(_bases) <= std::get<0>(other._bases);
    }

    constexpr std::enable_if_t<is_random_access_iterator_v<zipper>, bool>
    operator>=(const zipper &other) const {
      return std::get<0>(_bases) >= std::get<0>(other._bases);
    }

    constexpr std::enable_if_t<is_random_access_iterator_v<zipper>, reference>
    operator[](const difference_type n) const {
      return *(*this + n);
    }

    constexpr std::enable_if_t<is_random_access_iterator_v<zipper>,
                               difference_type>
    operator-(const zipper &other) const {
      return std::get<0>(_bases) - std::get<0>(other._bases);
    }

  private:
    std::tuple<Bs...> _bases;
  };

  template <typename... Bs> struct zip {
    template <typename... Ts>
    constexpr explicit zip(Ts &&... bases)
        : _bases{std::forward<Ts>(bases)...} {}

    constexpr bool empty() const {
      return std::empty(std::get<0>(_bases));
    }

    constexpr auto size() const {
      return std::size(std::get<0>(_bases));
    }

    constexpr auto begin() const {
      return std::apply(
          [](auto &&... bases) { return zipper{std::begin(bases)...}; },
          _bases);
    }

    constexpr auto cbegin() const {
      return std::apply(
          [](auto &&... bases) { return zipper{std::cbegin(bases)...}; },
          _bases);
    }

    constexpr auto end() const {
      return std::apply(
          [](auto &&... bases) { return zipper{std::end(bases)...}; }, _bases);
    }

    constexpr auto cend() const {
      return std::apply(
          [](auto &&... bases) { return zipper{std::cend(bases)...}; }, _bases);
    }

  private:
    std::tuple<Bs...> _bases;
  };

  template <typename... Ts> zip(Ts &&...)->zip<Ts...>;
} // namespace xsparse::range

namespace std {
  template <typename... Ts>
  void swap(xsparse::range::zip_proxy<Ts...> &&a,
            xsparse::range::zip_proxy<Ts...> &&b) {
    a.swap(b);
  }

  template <std::size_t Index, typename... Ts>
  constexpr decltype(auto)
  get(xsparse::range::zip_proxy<Ts...> &proxy) noexcept {
    return std::get<Index>(static_cast<std::tuple<Ts...> &>(proxy));
  }

  template <std::size_t Index, typename... Ts>
  constexpr decltype(auto)
  get(xsparse::range::zip_proxy<Ts...> &&proxy) noexcept {
    return std::get<Index>(static_cast<std::tuple<Ts...> &&>(proxy));
  }

  template <std::size_t Index, typename... Ts>
  constexpr decltype(auto)
  get(const xsparse::range::zip_proxy<Ts...> &proxy) noexcept {
    return std::get<Index>(static_cast<const std::tuple<Ts...> &>(proxy));
  }

  template <std::size_t Index, typename... Ts>
  constexpr decltype(auto)
  get(const xsparse::range::zip_proxy<Ts...> &&proxy) noexcept {
    return std::get<Index>(static_cast<const std::tuple<Ts...> &&>(proxy));
  }
} // namespace std