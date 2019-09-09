#include "test_util/test_transform.hpp"
#include <fftw3.h>
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
#include "mpi_util/mpi_communicator_handle.hpp"
#include "parameters/parameters.hpp"
#include "spfft/grid.hpp"
#include "spfft/transform.hpp"
#include "test_util/generate_indices.hpp"
#include "test_util/test_check_values.hpp"
#include "util/common_types.hpp"

class MPITransformTest : public TransformTest {
protected:
  MPITransformTest()
      : TransformTest(),
        comm_(MPI_COMM_WORLD),
        grid_(dimX_, dimY_, dimZ_, dimX_ * dimY_, dimZ_, std::get<1>(GetParam()), -1, comm_.get(),
              std::get<0>(GetParam())) {}

  auto comm_rank() -> SizeType override { return comm_.rank(); }

  auto comm_size() -> SizeType override { return comm_.size(); }

  auto grid() -> Grid& override { return grid_; }

  MPICommunicatorHandle comm_;
  Grid grid_;
};
TEST_P(MPITransformTest, ForwardUniformDistribution) {
  try {
    std::vector<double> zStickDistribution(comm_size(), 1.0);
    std::vector<double> xyPlaneDistribution(comm_size(), 1.0);
    test_forward_c2c(zStickDistribution, xyPlaneDistribution);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << comm_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(MPITransformTest, BackwardAllOneRank) {
  try {
    std::vector<double> zStickDistribution(comm_size(), 0.0);
    zStickDistribution[0] = 1.0;
    std::vector<double> xyPlaneDistribution(comm_size(), 0.0);
    xyPlaneDistribution[0] = 1.0;

    test_backward_c2c(zStickDistribution, xyPlaneDistribution);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << comm_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(MPITransformTest, ForwardAllOneRank) {
  try {
    std::vector<double> zStickDistribution(comm_size(), 0.0);
    zStickDistribution[0] = 1.0;
    std::vector<double> xyPlaneDistribution(comm_size(), 0.0);
    xyPlaneDistribution[0] = 1.0;

    test_forward_c2c(zStickDistribution, xyPlaneDistribution);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << comm_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(MPITransformTest, BackwardAllOneRankPerSide) {
  try {
    std::vector<double> zStickDistribution(comm_size(), 0.0);
    zStickDistribution[0] = 1.0;
    std::vector<double> xyPlaneDistribution(comm_size(), 0.0);
    xyPlaneDistribution[comm_size() - 1] = 1.0;

    test_backward_c2c(zStickDistribution, xyPlaneDistribution);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << comm_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(MPITransformTest, ForwardAllOneRankPerSide) {
  try {
    std::vector<double> zStickDistribution(comm_size(), 0.0);
    zStickDistribution[0] = 1.0;
    std::vector<double> xyPlaneDistribution(comm_size(), 0.0);
    xyPlaneDistribution[comm_size() - 1] = 1.0;

    test_forward_c2c(zStickDistribution, xyPlaneDistribution);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << comm_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(MPITransformTest, R2CUniformDistribution) {
  try {
    std::vector<double> xyPlaneDistribution(comm_size(), 1.0);
    test_r2c(xyPlaneDistribution);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << comm_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(MPITransformTest, R2COneRankAllPlanes) {
  try {
    std::vector<double> xyPlaneDistribution(comm_size(), 0.0);
    xyPlaneDistribution[0] = 1.0;
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
  switch (exchType) {
    case SpfftExchangeType::SPFFT_EXCH_BUFFERED: {
      name += "Buffered";
    } break;
    case SpfftExchangeType::SPFFT_EXCH_COMPACT_BUFFERED: {
      name += "CompactBuffered";
    } break;
    case SpfftExchangeType::SPFFT_EXCH_UNBUFFERED: {
      name += "Unbuffered";
    } break;
    default:
      name += "Default";
  }
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

INSTANTIATE_TEST_CASE_P(
    FullTest, MPITransformTest,
    ::testing::Combine(::testing::Values(SpfftExchangeType::SPFFT_EXCH_BUFFERED,
                                         SpfftExchangeType::SPFFT_EXCH_COMPACT_BUFFERED,
                                         SpfftExchangeType::SPFFT_EXCH_UNBUFFERED,
                                         SpfftExchangeType::SPFFT_EXCH_DEFAULT),
                       ::testing::Values(TEST_PROCESSING_UNITS),
                       ::testing::Values(1, 2, 11, 12, 13, 100),
                       ::testing::Values(1, 2, 11, 12, 13, 100),
                       ::testing::Values(1, 2, 11, 12, 13, 100), ::testing::Values(false)),
    param_type_names);

INSTANTIATE_TEST_CASE_P(CenteredIndicesTest, MPITransformTest,
                        ::testing::Combine(::testing::Values(SpfftExchangeType::SPFFT_EXCH_DEFAULT),
                                           ::testing::Values(TEST_PROCESSING_UNITS),
                                           ::testing::Values(1, 2, 11, 100),
                                           ::testing::Values(1, 2, 11, 100),
                                           ::testing::Values(1, 2, 11, 100),
                                           ::testing::Values(true)),
                        param_type_names);
