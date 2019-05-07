#pragma once

#include <iterator>
#include <tuple>
#include <type_traits>

namespace spectre {
  template <typename... Ts> struct zip_proxy : std::tuple<Ts...> {
    using std::tuple<Ts...>::tuple;
    using std::tuple<Ts...>::operator=;
  };

  template <typename... Rs> struct zip {
    using pointer = zip_proxy<typename Rs::pointer...>;
    using reference = zip_proxy<typename Rs::reference...>;
    using value_type = zip_proxy<typename Rs::value_type...>;
    using difference_type = std::common_type_t<typename Rs::difference_type...>;
    using size_type = std::common_type_t<typename Rs::size_type...>;

    struct iterator {
      using pointer = zip::pointer;
      using reference = zip::reference;
      using value_type = zip::value_type;
      using difference_type = zip::difference_type;
      using iterator_category = std::random_access_iterator_tag;

      constexpr iterator() = default;
      constexpr iterator(iterator &&) noexcept = default;
      constexpr iterator(iterator const &) noexcept = default;
      constexpr iterator &operator=(iterator &&) noexcept = default;
      constexpr iterator &operator=(iterator const &) = default;

      constexpr explicit iterator(typename Rs::iterator... bases)
          : _base{std::move(bases)...}
      {}

      constexpr iterator &operator++() noexcept
      {
        std::apply([](auto &&... bs) { (++bs, ...); }, _base);
        return *this;
      }

      constexpr iterator &operator--() noexcept
      {
        std::apply([](auto &&... bs) { (--bs, ...); }, _base);
        return *this;
      }

      constexpr iterator const operator++(int) noexcept
      {
        auto impl = [](auto &&... bs) { return iterator{bs++...}; };
        return std::apply(std::move(impl), _base);
      }

      constexpr iterator const operator--(int) noexcept
      {
        auto impl = [](auto &&... bs) { return iterator{bs--...}; };
        return std::apply(std::move(impl), _base);
      }

      constexpr bool operator==(iterator const &o) const noexcept
      {
        return std::get<0>(_base) == std::get<0>(o._base);
      }

      constexpr bool operator!=(iterator const &o) const noexcept
      {
        return std::get<0>(_base) != std::get<0>(o._base);
      }

      constexpr bool operator<=(iterator const &o) const noexcept
      {
        return std::get<0>(_base) <= std::get<0>(o._base);
      }

      constexpr bool operator>=(iterator const &o) const noexcept
      {
        return std::get<0>(_base) >= std::get<0>(o._base);
      }

      constexpr bool operator<(iterator const &o) const noexcept
      {
        return std::get<0>(_base) < std::get<0>(o._base);
      }

      constexpr bool operator>(iterator const &o) const noexcept
      {
        return std::get<0>(_base) > std::get<0>(o._base);
      }

      constexpr iterator operator+(difference_type const n) const noexcept
      {
        auto impl = [&n](auto &&... bs) { return iterator{bs + n...}; };
        return std::apply(std::move(impl), _base);
      }

      constexpr iterator operator-(difference_type const n) const noexcept
      {
        auto impl = [&n](auto &&... bs) { return iterator{bs - n...}; };
        return std::apply(std::move(impl), _base);
      }

      constexpr iterator &operator+=(difference_type const n) noexcept
      {
        std::apply([&n](auto &&... bs) { ((bs += n), ...); }, _base);
        return *this;
      }

      constexpr iterator &operator-=(difference_type const n) noexcept
      {
        std::apply([&n](auto &&... bs) { ((bs -= n), ...); }, _base);
        return *this;
      }

      constexpr reference operator*() const noexcept
      {
        auto impl = [](auto &&... bs) { return reference{*bs...}; };
        return std::apply(std::move(impl), _base);
      }

      constexpr reference operator[](difference_type const n) const noexcept
      {
        auto impl = [&n](auto &&... bs) { return reference{bs[n]...}; };
        return std::apply(std::move(impl), _base);
      }

      constexpr difference_type operator-(iterator const &o) const noexcept
      {
        auto diff = std::get<0>(_base) - std::get<0>(o._base);
        return static_cast<difference_type>(diff);
      }

    private:
      std::tuple<typename Rs::iterator...> _base;
    };

    constexpr zip() = default;
    constexpr zip(zip &&) noexcept = default;
    constexpr zip(zip const &) = default;
    constexpr zip &operator=(zip &&) noexcept = default;
    constexpr zip &operator=(zip const &) = default;

    constexpr explicit zip(Rs... bases) noexcept : _base{std::move(bases)...}
    {}

    constexpr bool empty() const noexcept
    {
      return std::get<0>(_base).empty();
    }

    constexpr size_type size() const noexcept
    {
      return std::get<0>(_base).size();
    }

    constexpr iterator begin() const noexcept
    {
      auto impl = [](auto &&... bs) { return iterator{bs.begin()...}; };
      return std::apply(std::move(impl), _base);
    }

    constexpr iterator end() const noexcept
    {
      auto impl = [](auto &&... bs) { return iterator{bs.end()...}; };
      return std::apply(std::move(impl), _base);
    }

    constexpr zip slice(size_type const a, size_type const b) const noexcept
    {
      auto impl = [&](auto &&... bs) { return zip{bs.slice(a, b)...}; };
      return std::apply(std::move(impl), _base);
    }

    constexpr reference operator[](size_type const n) const noexcept
    {
      auto impl = [&n](auto &&... bs) { return reference{bs[n]...}; };
      return std::apply(std::move(impl), _base);
    }

    constexpr reference front() const noexcept
    {
      auto impl = [](auto &&... bs) { return reference{bs.front()...}; };
      return std::apply(std::move(impl), _base);
    }

    constexpr reference back() const noexcept
    {
      auto impl = [](auto &&... bs) { return reference{bs.back()...}; };
      return std::apply(std::move(impl), _base);
    }

  private:
    std::tuple<Rs...> _base;
  };

  template <typename R> struct adjacent {
    template <typename T> using proxy = zip_proxy<T, T>;

    using pointer = proxy<typename R::pointer>;
    using reference = proxy<typename R::reference>;
    using value_type = proxy<typename R::value_type>;
    using difference_type = typename R::difference_type;
    using size_type = typename R::size_type;

    struct iterator {
      using pointer = adjacent::pointer;
      using reference = adjacent::reference;
      using value_type = adjacent::value_type;
      using difference_type = adjacent::difference_type;
      using iterator_category = std::random_access_iterator_tag;

      constexpr iterator() noexcept = default;
      constexpr iterator(iterator &&) noexcept = default;
      constexpr iterator(iterator const &) noexcept = default;
      constexpr iterator &operator=(iterator &&) noexcept = default;
      constexpr iterator &operator=(iterator const &) noexcept = default;

      constexpr explicit iterator(typename R::iterator base)
          : _base{std::move(base)}
      {}

      constexpr iterator &operator++() noexcept
      {
        return ++_base, *this;
      }

      constexpr iterator &operator--() noexcept
      {
        return --_base, *this;
      }

      constexpr iterator const operator++(int) noexcept
      {
        return iterator{_base++};
      }

      constexpr iterator const operator--(int) noexcept
      {
        return iterator{_base--};
      }

      constexpr bool operator==(iterator const &o) const noexcept
      {
        return _base == o._base;
      }

      constexpr bool operator!=(iterator const &o) const noexcept
      {
        return _base != o._base;
      }

      constexpr bool operator<=(iterator const &o) const noexcept
      {
        return _base <= o._base;
      }

      constexpr bool operator>=(iterator const &o) const noexcept
      {
        return _base >= o._base;
      }

      constexpr bool operator<(iterator const &o) const noexcept
      {
        return _base < o._base;
      }

      constexpr bool operator>(iterator const &o) const noexcept
      {
        return _base > o._base;
      }

      constexpr iterator operator+(difference_type const n) const noexcept
      {
        return iterator{_base + n};
      }

      constexpr iterator operator-(difference_type const n) const noexcept
      {
        return iterator{_base - n};
      }

      constexpr iterator &operator+=(difference_type const n) noexcept
      {
        return _base += n, *this;
      }

      constexpr iterator &operator-=(difference_type const n) noexcept
      {
        return _base -= n, *this;
      }

      constexpr reference operator*() const noexcept
      {
        auto temp = _base;
        return reference{*_base, *++temp};
      }

      constexpr reference operator[](difference_type const n) const noexcept
      {
        return reference{_base[n], _base[n + 1]};
      }

      constexpr difference_type operator-(iterator const &o) const noexcept
      {
        return _base - o._base;
      }

    private:
      typename R::iterator _base;
    };

    constexpr adjacent() noexcept = default;
    constexpr adjacent(adjacent &&) noexcept = default;
    constexpr adjacent(adjacent const &) noexcept = default;
    constexpr adjacent &operator=(adjacent &&) noexcept = default;
    constexpr adjacent &operator=(adjacent const &) noexcept = default;

    constexpr explicit adjacent(R base) noexcept : _base{std::move(base)}
    {}

    constexpr bool empty() const noexcept
    {
      return _base.size() > 1;
    }

    constexpr size_type size() const noexcept
    {
      auto temp = _base.size();
      return temp == 0 ? 0 : temp - 1;
    }

    constexpr iterator begin() const noexcept
    {
      return iterator{_base.begin()};
    }

    constexpr iterator end() const noexcept
    {
      auto temp = _base.end();
      return iterator{_base.empty() ? temp : temp - 1};
    }

    constexpr adjacent slice(size_type const a, size_type const b) const
        noexcept
    {
      return adjacent{_base.slice(a, b)};
    }

    constexpr reference operator[](size_type const n) const noexcept
    {
      return begin()[n];
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
    R _base;
  };
} // namespace spectre

namespace std {
  template <typename... Ts>
  struct tuple_size<::spectre::zip_proxy<Ts...>>
      : std::tuple_size<std::tuple<Ts...>> {};

  template <std::size_t I, typename... Ts>
  struct tuple_element<I, ::spectre::zip_proxy<Ts...>>
      : std::tuple_element<I, std::tuple<Ts...>> {};
} // namespace std
