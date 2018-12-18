#pragma once

#include <iterator>
#include <type_traits>

namespace xsparse::range {
  template <typename T>
  using iterator_category_t = typename std::iterator_traits<
      T>::iterator_category;

  template <typename T>
  using iterator_difference_t = typename std::iterator_traits<
      T>::difference_type;

  template <typename T>
  using iterator_value_t = typename std::iterator_traits<T>::value_type;

  template <typename T>
  using iterator_reference_t = typename std::iterator_traits<T>::reference;

  template <typename T>
  using iterator_pointer_t = typename std::iterator_traits<T>::pointer;

  template <typename T>
  using range_begin_t = decltype(std::begin(std::declval<T>()));

  template <typename T>
  using range_end_t = decltype(std::end(std::declval<T>()));

  template <typename T>
  constexpr inline bool is_input_iterator_v = std::is_base_of_v<
      std::input_iterator_tag, iterator_category_t<T>>;

  template <typename T>
  constexpr inline bool is_forward_iterator_v = std::is_base_of_v<
      std::forward_iterator_tag, iterator_category_t<T>>;

  template <typename T>
  constexpr inline bool is_bidirectional_iterator_v = std::is_base_of_v<
      std::bidirectional_iterator_tag, iterator_category_t<T>>;

  template <typename T>
  constexpr inline bool is_random_access_iterator_v = std::is_base_of_v<
      std::random_access_iterator_tag, iterator_category_t<T>>;
} // namespace xsparse::range