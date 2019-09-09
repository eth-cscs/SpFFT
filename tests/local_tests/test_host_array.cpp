#include <vector>
#include "gtest/gtest.h"
#include "memory/host_array.hpp"

using namespace spfft;

class HostArrayTest : public ::testing::Test {
protected:
  void SetUp() override {
    array_ = HostArray<int>(5);

    int count = 0;
    auto data_ptr = array_.data();
    for (SizeType i = 0; i < 5; ++i) {
      data_ptr[i] = ++count;
    }
  }

  HostArray<int> array_;
};

TEST_F(HostArrayTest, Iterators) {
  ASSERT_EQ(*array_.begin(), 1);
  ASSERT_EQ(*(array_.end() - 1), 5);
  int count = 0;
  for (auto& val : array_) {
    EXPECT_EQ(val, ++count);
  }
}

TEST_F(HostArrayTest, OperatorAccess) {
  int count = 0;
  ASSERT_EQ(array_.size(), 5);
  for (SizeType i = 0; i < array_.size(); ++i) {
    ASSERT_EQ(array_[i], ++count);
  }
  count = 0;
  for (SizeType i = 0; i < array_.size(); ++i) {
    ASSERT_EQ(array_(i), ++count);
  }
}

TEST_F(HostArrayTest, Accessors) {
  ASSERT_EQ(array_.front(), 1);
  ASSERT_EQ(array_.back(), 5);
}
