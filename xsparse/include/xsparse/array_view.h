#pragma once

#include "array_traits.h"
#include "range_span.h"
#include "range_zip.h"
#include <tuple>

namespace xsparse {
  template <typename Ty, typename... Cxs> struct coo_view : coo_array_tag {
    using coord_type = std::tuple<Cxs...>;
    using value_type = Ty;
    using shape_type = std::tuple<Cxs...>;
    using index_type = typename range::span<value_type>::index_type;

    constexpr static auto ndims = sizeof...(Cxs);

    constexpr coo_view() = default;

    constexpr coo_view(shape_type shape)
        : _shape{std::move(shape)}
        , _values{}
        , _coords{} {}

    template <typename U, typename... Us>
    constexpr coo_view(shape_type shape, U &values, Us &... coords)
        : _shape{std::move(shape)}
        , _values{range::span{values}}
        , _coords{range::span{coords}...} {}

    constexpr bool empty() const {
      return std::empty(_values);
    }

    constexpr index_type size() const {
      return std::size(_values);
    }

    constexpr shape_type shape() const {
      return _shape;
    }

    constexpr auto values() -> range::span<value_type> {
      return _values;
    }

    constexpr auto values() const -> range::span<const value_type> {
      return _values;
    }

    constexpr auto cvalues() const -> range::span<const value_type> {
      return _values;
    }

    constexpr auto coords() -> range::zip<range::span<Cxs>...> {
      return range::zip{_coords};
    }

    constexpr auto coords() const -> range::zip<range::span<const Cxs>...> {
      return range::zip{_coords};
    }

    constexpr auto ccoords() const -> range::zip<range::span<const Cxs>...> {
      return coords();
    }

    constexpr auto sparse() {
      return range::zip{coords(), values()};
    }

    constexpr auto sparse() const {
      return range::zip{coords(), values()};
    }

    constexpr auto csparse() const {
      return sparse();
    }

  private:
    shape_type _shape;
    range::span<Ty> _values;
    std::tuple<range::span<Cxs>...> _coords;
  };

  template <typename U, typename... Us>
  coo_view(std::tuple<typename Us::value_type...>, U &values, Us &... coords)
      ->coo_view<typename U::value_type, typename Us::value_type...>;
} // namespace xsparse