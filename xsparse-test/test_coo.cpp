#include <array_view.h>
#include <gtest/gtest.h>
#include <type_traits>
#include <vector>

using xsparse::coo_view;

TEST(coo_test, dummy) {
  auto row = std::vector{1, 6, 7, 8};
  auto col = std::vector{1, 2, 3, 4};
  auto val = std::vector{1, 2, 4, 8};

  auto array = coo_view{std::tuple{9, 5}, val, row, col};
}
