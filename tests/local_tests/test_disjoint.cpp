#include <vector>
#include "gtest/gtest.h"
#include "memory/array_view_utility.hpp"
#include "memory/host_array.hpp"
#include "memory/host_array_view.hpp"

using namespace spfft;

class DisjointTest : public ::testing::Test {
protected:
  void SetUp() override { array_ = HostArray<int>(100); }

  HostArray<int> array_;
};

TEST_F(DisjointTest, dim1AndDim1) {
  {
    auto view1 = create_1d_view(array_, 0, 10);
    auto view2 = create_1d_view(array_, 0, 10);
    EXPECT_FALSE(disjoint(view1, view2));
  }
  {
    auto view1 = create_1d_view(array_, 0, 10);
    auto view2 = create_1d_view(array_, 5, 10);
    EXPECT_FALSE(disjoint(view1, view2));
  }
  {
    auto view1 = create_1d_view(array_, 0, 10);
    auto view2 = create_1d_view(array_, 10, 10);
    EXPECT_TRUE(disjoint(view1, view2));
  }
}

TEST_F(DisjointTest, dim1AndDim2) {
  {
    auto view1 = create_1d_view(array_, 0, 10);
    auto view2 = create_2d_view(array_, 0, 2, 5);
    EXPECT_FALSE(disjoint(view1, view2));
  }
  {
    auto view1 = create_1d_view(array_, 0, 10);
    auto view2 = create_2d_view(array_, 5, 2, 5);
    EXPECT_FALSE(disjoint(view1, view2));
  }
  {
    auto view1 = create_1d_view(array_, 0, 10);
    auto view2 = create_2d_view(array_, 10, 5, 2);
    EXPECT_TRUE(disjoint(view1, view2));
  }
}

TEST_F(DisjointTest, dim1AndDim3) {
  {
    auto view1 = create_1d_view(array_, 0, 10);
    auto view2 = create_3d_view(array_, 0, 2, 2, 2);
    EXPECT_FALSE(disjoint(view1, view2));
  }
  {
    auto view1 = create_1d_view(array_, 0, 10);
    auto view2 = create_3d_view(array_, 5, 2, 2, 2);
    EXPECT_FALSE(disjoint(view1, view2));
  }
  {
    auto view1 = create_1d_view(array_, 0, 10);
    auto view2 = create_3d_view(array_, 10, 5, 2, 2);
    EXPECT_TRUE(disjoint(view1, view2));
  }
}

TEST_F(DisjointTest, dim2AndDim3) {
  {
    auto view1 = create_2d_view(array_, 0, 2, 3);
    auto view2 = create_3d_view(array_, 0, 2, 3, 2);
    EXPECT_FALSE(disjoint(view1, view2));
  }
  {
    auto view1 = create_2d_view(array_, 0, 2, 3);
    auto view2 = create_3d_view(array_, 5, 2, 2, 2);
    EXPECT_FALSE(disjoint(view1, view2));
  }
  {
    auto view1 = create_2d_view(array_, 0, 2, 3);
    auto view2 = create_3d_view(array_, 6, 5, 2, 2);
    EXPECT_TRUE(disjoint(view1, view2));
  }
}

TEST_F(DisjointTest, dim3AndDim3) {
  {
    auto view1 = create_3d_view(array_, 0, 2, 3, 4);
    auto view2 = create_3d_view(array_, 0, 2, 3, 2);
    EXPECT_FALSE(disjoint(view1, view2));
  }
  {
    auto view1 = create_3d_view(array_, 0, 2, 3, 4);
    auto view2 = create_3d_view(array_, 5, 2, 2, 2);
    EXPECT_FALSE(disjoint(view1, view2));
  }
  {
    auto view1 = create_3d_view(array_, 0, 2, 3, 2);
    auto view2 = create_3d_view(array_, 12, 5, 2, 2);
    EXPECT_TRUE(disjoint(view1, view2));
  }
}

TEST_F(DisjointTest, DifferentValueTypes) {
  auto view1 = create_3d_view(array_, 0, 2, 3, 4);
  auto view2 =
      HostArrayView3D<long long>(reinterpret_cast<long long*>(array_.data()), 2, 3, 4, false);
  EXPECT_FALSE(disjoint(view1, view2));
}
