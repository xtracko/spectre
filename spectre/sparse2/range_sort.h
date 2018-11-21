#pragma once

#include "range_span.h"
#include <algorithm>

namespace range {
  namespace detail {
    template <unsigned Small, typename Rn, typename Cp>
    void sort_impl(Rn&& range, Cp&& comparator) {
      if (range.size() >= Small) {
        const auto mid = range.size() / 2;

#pragma omp taskgroup
        {
#pragma omp task shared(range, comparator) untied if (range.size() >= (1 << 14))
          sort_impl<Small>(range.first(mid), comparator);
#pragma omp task shared(range, comparator) untied if (range.size() >= (1 << 14))
          sort_impl<Small>(range.last(mid), comparator);
#pragma omp taskyield
        }

        std::inplace_merge(range.begin(), range.begin() + mid, range.end(),
                           comparator);
      } else {
        std::sort(range.begin(), range.end(), comparator);
      }
    }
  } // namespace detail

  template <typename Rn, typename Cp>
  decltype(auto) sort(Rn&& range, Cp&& comparator) {
#pragma omp parallel
#pragma omp single
    detail::sort_impl<32>(std::forward<Rn>(range),
                          std::forward<Cp>(comparator));
    return std::forward<Rn>(range);
  }

  template <typename Rn> decltype(auto) sort(Rn&& range) {
    return sort(std::forward<Rn>(range), std::less{});
  }
} // namespace range