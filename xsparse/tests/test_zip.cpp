#include <gtest/gtest.h>
#include <xsparse/range_span.h>
#include <xsparse/range_zip.h>

using xsparse::range::span;
using xsparse::range::zip;

TEST(zip_test, noneven_subspans) {
  auto a = std::array{0, 1};
  auto b = std::array{0, 1, 2};
  auto c = zip{span{a}, span{b}};

  EXPECT_FALSE(c.empty());
  EXPECT_EQ(c.size(), 2);

  EXPECT_NE(c.begin(), c.end());
  EXPECT_EQ(std::distance(c.begin(), c.end()), 2);
}

TEST(zip_test, random_access_iterator) {
  const auto a = std::array{0, 1, 2, 3};
  const auto b = std::array{'a', 'b', 'c', 'd'};
  const auto c = zip{span{a}, span{b}};

  ASSERT_FALSE(c.empty());
  ASSERT_EQ(c.size(), 4);

  auto i = c.begin();
  auto j = c.end();

  using iterator = std::decay_t<decltype(i)>;
  using category = typename std::iterator_traits<iterator>::iterator_category;
  EXPECT_TRUE((std::is_base_of_v<std::random_access_iterator_tag, category>));

  EXPECT_EQ(i, i);
  EXPECT_NE(i, j);

  EXPECT_LT(i, j);
  EXPECT_LE(i, j);
  EXPECT_GT(j, i);
  EXPECT_GE(j, j);

  EXPECT_NE(i + 3, j);
  EXPECT_LT(i + 3, j);
  EXPECT_LE(i + 3, j);

  EXPECT_EQ(i + 4, j);
  EXPECT_LE(i + 4, j);

  EXPECT_EQ(i[0], (std::tuple{0, 'a'}));
  EXPECT_EQ(i[1], (std::tuple{1, 'b'}));
  EXPECT_EQ(i[2], (std::tuple{2, 'c'}));
  EXPECT_EQ(i[3], (std::tuple{3, 'd'}));

  EXPECT_EQ(i - i, 0);
  EXPECT_EQ(j - j, 0);

  EXPECT_EQ(j - i, 4);
  EXPECT_EQ(i - j, -4);

  EXPECT_EQ(i + 4, c.end());
  EXPECT_EQ(j - 4, c.begin());
}

TEST(zip_test, forward_iteration) {
  auto a = std::array{0, 1, 2, 3};
  auto b = std::array{'a', 'b', 'c', 'd'};
  auto c = zip{span{a}, span{b}};

  ASSERT_FALSE(c.empty());
  ASSERT_EQ(c.size(), 4);

  auto i = c.begin();
  auto j = c.end();

  using iterator = std::decay_t<decltype(i)>;
  using category = typename std::iterator_traits<iterator>::iterator_category;
  EXPECT_TRUE((std::is_base_of_v<std::forward_iterator_tag, category>));

  ASSERT_NE(i, j);
  EXPECT_EQ(*i, (std::tuple{0, 'a'}));
  EXPECT_EQ(*++i, (std::tuple{1, 'b'}));
  EXPECT_EQ(*++i, (std::tuple{2, 'c'}));
  EXPECT_EQ(*++i, (std::tuple{3, 'd'}));
  EXPECT_EQ(++i, j);
}

TEST(zip_test, proxy_swap) {
  auto a = std::array{1, 2};
  auto b = std::array{'a', 'b'};

  auto c = zip{span{a}, span{b}};

  auto i1 = c.begin();
  auto i2 = ++c.begin();

  using std::swap;
  swap(*i1, *i2);

  EXPECT_EQ(a[0], 2);
  EXPECT_EQ(a[1], 1);

  EXPECT_EQ(b[0], 'b');
  EXPECT_EQ(b[1], 'a');
}

TEST(zip_test, class_template_deduction) {
  auto array = std::vector<int>{};
  const auto carray = std::vector<int>{};

  using zip_by_reference = decltype(zip{array});
  using zip_by_const_reference = decltype(zip{carray});

  ::testing::StaticAssertTypeEq<zip_by_reference, zip<std::vector<int> &>>();
  ::testing::StaticAssertTypeEq<zip_by_const_reference,
                                zip<const std::vector<int> &>>();

  using zip_by_rvalue = decltype(zip{std::move(array)});
  using zip_by_const_rvalue = decltype(zip{std::move(carray)});

  ::testing::StaticAssertTypeEq<zip_by_rvalue, zip<std::vector<int>>>();
  ::testing::StaticAssertTypeEq<zip_by_const_rvalue,
                                zip<const std::vector<int>>>();
}
