#include <algorithm>
#include <memory>
#include <random>
#include <tuple>
#include <utility>
#include <vector>
#include "gtest/gtest.h"
#include "memory/array_view_utility.hpp"
#include "memory/host_array.hpp"
#include "memory/host_array_view.hpp"
#include "parameters/parameters.hpp"
#include "spfft/grid.hpp"
#include "spfft/transform.hpp"
#include "test_util/generate_indices.hpp"
#include "test_util/test_check_values.hpp"
#include "test_util/test_transform.hpp"
#include "util/common_types.hpp"

class TestLocalTransform : public TransformTest {
protected:
  TestLocalTransform()
      : TransformTest(), grid_(dimX_, dimY_, dimZ_, dimX_ * dimY_, std::get<1>(GetParam()), -1) {}

  auto grid() -> Grid& override { return grid_; }

  Grid grid_;
};
TEST_P(TestLocalTransform, ForwardC2C) {
  try {
    std::vector<double> zStickDistribution(comm_size(), 1.0);
    std::vector<double> xyPlaneDistribution(comm_size(), 1.0);
    test_forward_c2c(zStickDistribution, xyPlaneDistribution);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << comm_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(TestLocalTransform, BackwardC2C) {
  try {
    std::vector<double> zStickDistribution(comm_size(), 1.0);
    std::vector<double> xyPlaneDistribution(comm_size(), 1.0);

    test_backward_c2c(zStickDistribution, xyPlaneDistribution);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << comm_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(TestLocalTransform, R2C) {
  try {
    std::vector<double> xyPlaneDistribution(comm_size(), 1.0);
    test_r2c(xyPlaneDistribution);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << comm_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

// Show exchange name instead of enum value for test output
static auto param_type_names(
    const ::testing::TestParamInfo<
        std::tuple<SpfftExchangeType, SpfftProcessingUnitType, int, int, int, bool>>& info)
    -> std::string {
  const auto exchType = std::get<0>(info.param);
  const auto procType = std::get<1>(info.param);
  std::string name;
  switch (procType) {
    case SpfftProcessingUnitType::SPFFT_PU_HOST: {
      name += "Host";
    } break;
    case SpfftProcessingUnitType::SPFFT_PU_GPU: {
      name += "GPU";
    } break;
    default: { name += "Host+GPU"; }
  }
  name += "Size";
  name += std::to_string(std::get<2>(info.param));
  name += "x";
  name += std::to_string(std::get<3>(info.param));
  name += "x";
  name += std::to_string(std::get<4>(info.param));
  return name;
}

// instantiate tests with parameters
#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
#define TEST_PROCESSING_UNITS \
  SpfftProcessingUnitType::SPFFT_PU_HOST, SpfftProcessingUnitType::SPFFT_PU_GPU
#else
#define TEST_PROCESSING_UNITS SpfftProcessingUnitType::SPFFT_PU_HOST
#endif

INSTANTIATE_TEST_CASE_P(FullTest, TestLocalTransform,
                        ::testing::Combine(::testing::Values(SpfftExchangeType::SPFFT_EXCH_DEFAULT),
                                           ::testing::Values(TEST_PROCESSING_UNITS),
                                           ::testing::Values(1, 2, 11, 12, 13, 100),
                                           ::testing::Values(1, 2, 11, 12, 13, 100),
                                           ::testing::Values(1, 2, 11, 12, 13, 100),
                                           ::testing::Values(false)),
                        param_type_names);

INSTANTIATE_TEST_CASE_P(CenteredIndicesTest, TestLocalTransform,
                        ::testing::Combine(::testing::Values(SpfftExchangeType::SPFFT_EXCH_DEFAULT),
                                           ::testing::Values(TEST_PROCESSING_UNITS),
                                           ::testing::Values(1, 2, 11, 100),
                                           ::testing::Values(1, 2, 11, 100),
                                           ::testing::Values(1, 2, 11, 100),
                                           ::testing::Values(true)),
                        param_type_names);
