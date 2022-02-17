#include <unordered_set>
#include "gtest/gtest.h"
#include "fft/fftw_plan_1d.hpp"


TEST(FFTWPropHashTest, Unique) {
  std::unordered_set<std::tuple<bool, int, int>,  spfft::FFTWPropHash> set;

  int maxAlignment = 1024;

  for (int inPlace = 0; inPlace < 2; ++inPlace) {
    for (int i = 0 ;i < maxAlignment; ++i) {
      for (int j = 0; j < maxAlignment; ++j) {
        set.emplace(inPlace, i, j);
      }
    }
  }

  EXPECT_EQ(static_cast<std::size_t>(maxAlignment) * static_cast<std::size_t>(maxAlignment) * 2,
            set.size());
}
