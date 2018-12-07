#pragma once

#include "array_traits.h"
#include "range_span.h"
#include "range_zip.h"
#include <pybind11/numpy.h>

template <typename U>
using pyarray =
    pybind11::array_t<U, pybind11::array::c_style | pybind11::array::forcecast>;

template <typename Ty, typename... Cxs> struct pycoo : coo_array_tag {
  using coord_type = std::tuple<Cxs...>;
  using value_type = Ty;
  using shape_type = std::tuple<Cxs...>;
  using index_type = std::ptrdiff_t;

  constexpr static auto ndims = sizeof...(Cxs);

  constexpr pycoo() = default;

  constexpr pycoo(shape_type shape)
      : _shape{std::move(shape)}
      , _values{}
      , _coords{} {
  }

  constexpr pycoo(shape_type shape, std::size_t size)
      : _shape{std::move(shape)}
      , _values{size}
      , _coords{pyarray<Cxs>{size}...} {
  }

  constexpr pycoo(shape_type shape, pyarray<Ty> values, pyarray<Cxs>... coords)
      : _shape{std::move(shape)}
      , _values{std::move(values)}
      , _coords{std::move(coords)...} {
  }

  constexpr bool empty() const {
    return size() == 0;
  }

  constexpr index_type size() const {
    return static_cast<index_type>(_values.size());
  }

  constexpr shape_type shape() const {
    return _shape;
  }

  constexpr auto values() {
    return range::span<value_type>{_values.mutable_data(0), _values.size()};
  }

  constexpr auto values() const {
    return range::span<const value_type>{_values.data(0), _values.size()};
  }

  constexpr auto cvalues() const {
    return cvalues();
  }

  constexpr auto coords() {
    return std::apply(
        [](auto&&... coords) {
          return range::zip{
              range::span<Cxs>{coords.mutable_data(0), coords.size()}...};
        },
        _coords);
  }

  constexpr auto coords() const {
    return std::apply(
        [](auto&&... coords) {
          return range::zip{
              range::span<const Cxs>{coords.data(0), coords.size()}...};
        },
        _coords);
  }

  constexpr auto ccoords() const {
    return coords();
  }

  constexpr auto sparse() {
    return range::zip{coords(), values()};
  }

  constexpr auto sparse() const {
    return range::zip{coords(), values()};
  }

  constexpr auto csparse() const {
    return range::zip{ccoords(), cvalues()};
  }

private:
  shape_type _shape;
  pyarray<value_type> _values;
  std::tuple<pyarray<Cxs>...> _coords;

  friend pybind11::detail::type_caster<pycoo>;
};

namespace pybind11::detail {
  template <typename Ty, typename Ix, typename Jx>
  struct type_caster<pycoo<Ty, Ix, Jx>> {
    using type = pycoo<Ty, Ix, Jx>;

    PYBIND11_TYPE_CASTER(type,
                         _("scipy.sparse.coo_matrix[") +
                             npy_format_descriptor<Ix>::name() + _(", ") +
                             npy_format_descriptor<Jx>::name() + _(", ") +
                             npy_format_descriptor<Ty>::name() + _("]"));

    bool load(handle source, bool) {
      auto scipy_type = module::import("scipy.sparse").attr("coo_matrix");

      if (!isinstance(source, scipy_type))
        return false;

      auto row = object{source.attr("row")};
      auto col = object{source.attr("col")};
      auto val = object{source.attr("data")};
      auto shape = object{source.attr("shape")};

      if (!isinstance<pyarray<Ix>>(row) || !isinstance<pyarray<Jx>>(col) ||
          !isinstance<pyarray<Ty>>(val) || !isinstance<pybind11::tuple>(shape))
        return false;

      auto casted_row = pyarray<Ix>{std::move(row)};
      auto casted_col = pyarray<Jx>{std::move(col)};
      auto casted_val = pyarray<Ty>{std::move(val)};
      auto casted_shape = pybind11::cast<std::tuple<Ix, Jx>>(std::move(shape));

      if (!casted_row || !casted_col || !casted_val)
        return false;

      value = type{std::move(casted_shape), std::move(casted_val),
                   std::move(casted_row), std::move(casted_col)};
      return true;
    }

    static handle cast(const type& source, return_value_policy, handle) {
      auto scipy_type = module::import("scipy.sparse").attr("coo_matrix");

      return scipy_type(std::tuple{source._values, source._coords},
                        source._shape)
          .release();
    }
  };
} // namespace pybind11::detail
