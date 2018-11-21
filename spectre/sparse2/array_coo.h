#pragma once

#include "array_traits.h"
#include "range_span.h"
#include "range_zip.h"
#include <tuple>

template <typename Ix, typename Jx, typename Ty> struct coo_view {
  using tag = coo_array_tag;
  using ix_type = Ix;
  using jx_type = Jx;
  using coord_type = std::tuple<ix_type, jx_type>;
  using value_type = Ty;
  using index_type = typename range::span<value_type>::index_type;

  constexpr coo_view() = default;

  constexpr coo_view(range::span<ix_type> axs,
                     range::span<jx_type> bxs,
                     range::span<value_type> data,
                     std::tuple<ix_type, jx_type> shape)
      : _ixs{std::move(axs)}
      , _jxs{std::move(bxs)}
      , _values{std::move(data)}
      , _shape{std::move(shape)} {
  }

  constexpr bool empty() const {
    return std::empty(_values);
  }

  constexpr index_type size() const {
    return std::size(_values);
  }

  constexpr auto coords() const
      -> range::zip<range::span<ix_type>, range::span<jx_type>> {
    return range::zip{_ixs, _jxs};
  }

  constexpr auto values() const -> range::span<value_type> {
    return _values;
  }

  constexpr std::tuple<ix_type, jx_type> shape() const {
    return _shape;
  }

  constexpr auto sparse() const
      -> range::zip<range::zip<range::span<ix_type>, range::span<jx_type>>,
                    range::span<value_type>> {
    return range::zip{coords(), values()};
  }

private:
  range::span<ix_type> _ixs;
  range::span<jx_type> _jxs;
  range::span<value_type> _values;
  std::tuple<ix_type, jx_type> _shape;
};