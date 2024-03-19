#include <fftw3.h>

#include <algorithm>
#include <memory>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "gtest_mpi.hpp"
#include "memory/array_view_utility.hpp"
#include "memory/host_array.hpp"
#include "memory/host_array_view.hpp"
#include "mpi_util/mpi_communicator_handle.hpp"
#include "parameters/parameters.hpp"
#include "spfft/spfft.hpp"
#include "test_util/generate_indices.hpp"
#include "test_util/test_transform.hpp"
#include "util/common_types.hpp"

TEST(MPIMultiTransformTest, BackwardsForwards) {
  GTEST_MPI_GUARD
  try {
    MPICommunicatorHandle comm(MPI_COMM_WORLD);
    const std::vector<double> zStickDistribution(comm.size(), 1.0);
    const std::vector<double> xyPlaneDistribution(comm.size(), 1.0);

    const int dimX = comm.size() * 10;
    const int dimY = comm.size() * 11;
    const int dimZ = comm.size() * 12;

    const int numTransforms = 3;

    std::mt19937 randGen(42);
    const auto valueIndicesPerRank =
        create_value_indices(randGen, zStickDistribution, 0.7, 0.7, dimX, dimY, dimZ, false);
    const int numLocalXYPlanes =
        calculate_num_local_xy_planes(comm.rank(), dimZ, xyPlaneDistribution);

    const auto& localIndices = valueIndicesPerRank[comm.rank()];
    const int numValues = localIndices.size() / 3;
    std::vector<std::vector<std::complex<double>>> freqValuesPerTrans(
        numTransforms, std::vector<std::complex<double>>(numValues));

    std::vector<double*> freqValuePtr;
    for (auto& values : freqValuesPerTrans) {
      freqValuePtr.push_back(reinterpret_cast<double*>(values.data()));
    }

    // set frequency values to constant for each transform
    for (std::size_t i = 0; i < freqValuesPerTrans.size(); ++i) {
      for (auto& val : freqValuesPerTrans[i]) {
        val = std::complex<double>(i, i);
      }
    }

    std::vector<Transform> transforms;

    // create first transforms
    transforms.push_back(Grid(dimX, dimY, dimZ, dimX * dimY, numLocalXYPlanes, SPFFT_PU_HOST, -1,
                              comm.get(), SPFFT_EXCH_DEFAULT)
                             .create_transform(SPFFT_PU_HOST, SPFFT_TRANS_C2C, dimX, dimY, dimZ,
                                               numLocalXYPlanes, numValues, SPFFT_INDEX_TRIPLETS,
                                               localIndices.data()));
    // clone first transform
    for (int i = 1; i < numTransforms; ++i) {
      transforms.push_back(transforms.front().clone());
    }

    std::vector<SpfftProcessingUnitType> processingUnits(numTransforms, SPFFT_PU_HOST);
    std::vector<SpfftScalingType> scalingTypes(numTransforms, SPFFT_NO_SCALING);

    // backward
    multi_transform_backward(numTransforms, transforms.data(), freqValuePtr.data(),
                             processingUnits.data());

    // forward
    multi_transform_forward(numTransforms, transforms.data(), processingUnits.data(),
                            freqValuePtr.data(), scalingTypes.data());

    // check all values
    for (std::size_t i = 0; i < freqValuesPerTrans.size(); ++i) {
      const auto targetValue = std::complex<double>(i * dimX * dimY * dimZ, i * dimX * dimY * dimZ);
      for (auto& val : freqValuesPerTrans[i]) {
        ASSERT_NEAR(targetValue.real(), val.real(), 1e-8);
        ASSERT_NEAR(targetValue.imag(), val.imag(), 1e-8);
      }
    }

  } catch (const std::exception& e) {
    std::cout << "ERROR: " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}
