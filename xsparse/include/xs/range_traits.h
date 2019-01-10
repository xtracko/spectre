#include <iterator>
#include <type_traits>

namespace xs::range::detail_is_iterable {
  using std::begin;
  using std::end;

  template <typename Range, typename = std::void_t<>>
  struct is_iterable_impl : std::false_type {};

  template <typename Range>
  struct is_iterable_impl<Range,
                          std::void_t<decltype(begin(std::declval<Range>())),
                                      decltype(end(std::declval<Range>()))>>
      : std::true_type {};
} // namespace xs::range::detail_is_iterable

namespace xs::range {
  template <typename Iter>
  using iter_value_t = typename std::iterator_traits<Iter>::value_type;

  template <typename Iter>
  using iter_pointer_t = typename std::iterator_traits<Iter>::pointer;

  template <typename Iter>
  using iter_reference_t = typename std::iterator_traits<Iter>::reference;

  template <typename Iter>
  using iter_difference_t =
      typename std::iterator_traits<Iter>::difference_type;

  template <typename Iter>
  using iter_category_t =
      typename std::iterator_traits<Iter>::iterator_category;

  template <typename Iter>
  constexpr inline bool is_input_iterator =
      std::is_base_of_v<std::input_iterator_tag, iter_category_t<Iter>>;

  template <typename Iter>
  constexpr inline bool is_forward_iterator =
      std::is_base_of_v<std::forward_iterator_tag, iter_category_t<Iter>>;

  template <typename Iter>
  constexpr inline bool is_bidirectional_iterator =
      std::is_base_of_v<std::bidirectional_iterator_tag, iter_category_t<Iter>>;

  template <typename Iter>
  constexpr inline bool is_random_access_iterator =
      std::is_base_of_v<std::random_access_iterator_tag, iter_category_t<Iter>>;

  template <typename Iter, typename Type>
  using enable_input_iterator_t =
      std::enable_if_t<is_input_iterator<Iter>, Type>;

  template <typename Iter, typename Type>
  using enable_forward_iterator_t =
      std::enable_if_t<is_forward_iterator<Iter>, Type>;

  template <typename Iter, typename Type>
  using enable_bidirectional_iterator_t =
      std::enable_if_t<is_bidirectional_iterator<Iter>, Type>;

  template <typename Iter, typename Type>
  using enable_random_access_iterator_t =
      std::enable_if_t<is_random_access_iterator<Iter>, Type>;

  template <typename Type>
  constexpr inline bool is_iterable =
      detail_is_iterable::is_iterable_impl<Type>::value;
} // namespace xs::range