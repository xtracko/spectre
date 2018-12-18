#include <gtest/gtest.h>
#include <xsparse/range_span.h>

using xsparse::range::span;

TEST(span_test, empty) {
  auto view = span<float>{};

  EXPECT_EQ(view.size(), 0);
  EXPECT_TRUE(view.empty());

  EXPECT_EQ(view.begin(), nullptr);
  EXPECT_EQ(view.cbegin(), nullptr);

  EXPECT_EQ(view.end(), nullptr);
  EXPECT_EQ(view.cend(), nullptr);
}

TEST(span_test, nonempty) {
  auto data = std::array{0, 1, 2, 3};
  auto view = span<int>{data};

  EXPECT_EQ(view.size(), 4);
  EXPECT_FALSE(view.empty());

  EXPECT_EQ(view.begin(), data.data());
  EXPECT_EQ(view.cbegin(), data.data());

  EXPECT_EQ(view.end(), data.data() + 4);
  EXPECT_EQ(view.cend(), data.data() + 4);

  EXPECT_EQ(view.first(0).begin(), data.data());
  EXPECT_EQ(view.first(1).begin(), data.data());

  EXPECT_EQ(view.first(0).end(), data.data());
  EXPECT_EQ(view.first(1).end(), data.data() + 1);

  EXPECT_EQ(view.last(0).begin(), data.data() + 4);
  EXPECT_EQ(view.last(1).begin(), data.data() + 3);

  EXPECT_EQ(view.last(0).end(), data.data() + 4);
  EXPECT_EQ(view.last(1).end(), data.data() + 4);

  EXPECT_EQ(view.subspan(1, 1).begin(), data.data() + 1);
  EXPECT_EQ(view.subspan(1, 1).end(), data.data() + 2);
}

TEST(span_test, deductible) {
  auto a = std::array{0, 1};
  const auto b = std::array{'a', 'b'};

  ::testing::StaticAssertTypeEq<decltype(span(a)), span<int>>();
  ::testing::StaticAssertTypeEq<decltype(span(b)), span<const char>>();
}
