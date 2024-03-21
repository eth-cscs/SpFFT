#include <fftw3.h>

#include <algorithm>
#include <memory>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "gtest_mpi.hpp"
#include "memory/array_view_utility.hpp"
#include "memory/host_array.hpp"
#include "memory/host_array_view.hpp"
#include "mpi_util/mpi_communicator_handle.hpp"
#include "parameters/parameters.hpp"
#include "transpose/transpose_mpi_buffered_host.hpp"
#include "transpose/transpose_mpi_compact_buffered_host.hpp"
#include "transpose/transpose_mpi_unbuffered_host.hpp"
#include "util/common_types.hpp"

using namespace spfft;

class TransposeTest : public ::testing::Test {
protected:
  void SetUp() override {
    comm_ = MPICommunicatorHandle(MPI_COMM_WORLD);

    SizeType dimX = 2 * comm_.size();
    SizeType dimY = 3 * comm_.size();
    SizeType dimZ = 4 * comm_.size();

    // create memory space
    array1_ = HostArray<std::complex<double>>(dimX * dimY * dimZ, std::complex<double>(1.0, 1.0));
    array2_ = HostArray<std::complex<double>>(dimX * dimY * dimZ, std::complex<double>(1.0, 1.0));
    fullArray_ = HostArray<std::complex<double>>(dimX * dimY * dimZ);

    // plane split between ranks
    const SizeType numLocalXYPlanes =
        (dimZ / comm_.size()) + (comm_.rank() == comm_.size() - 1 ? dimZ % comm_.size() : 0);
    const SizeType localXYPlaneOffset = (dimZ / comm_.size()) * comm_.rank();

    // create all indices the same way (random generator must be equally initialized)
    std::mt19937 sharedRandGen(42);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    std::uniform_int_distribution<int> rankSelector(0, comm_.size() - 1);

    std::vector<int> indexTriplets;
    indexTriplets.reserve(dimX * dimY * dimZ);
    for (int x = 0; x < static_cast<int>(dimX); ++x) {
      for (int y = 0; y < static_cast<int>(dimY); ++y) {
        // create sparse z stick distribution
        if (dis(sharedRandGen) < 0.5 &&
            rankSelector(sharedRandGen) == static_cast<int>(comm_.size())) {
          for (int z = 0; z < static_cast<int>(dimY); ++z) {
            indexTriplets.push_back(x);
            indexTriplets.push_back(y);
            indexTriplets.push_back(z);
          }
        }
      }
    }

    paramPtr_.reset(new Parameters(comm_, SPFFT_TRANS_C2C, dimX, dimY, dimZ, numLocalXYPlanes,
                                   indexTriplets.size() / 3, SPFFT_INDEX_TRIPLETS,
                                   indexTriplets.data()));

    // initialize random z-stick data
    auto fullView = create_3d_view(fullArray_, 0, dimX, dimY, dimZ);
    auto freqView = create_2d_view(array1_, 0, paramPtr_->num_z_sticks(comm_.rank()), dimZ);

    for (SizeType r = 0; r < comm_.size(); ++r) {
      for (const auto& stickIdx : paramPtr_->z_stick_xy_indices(r)) {
        const auto x = stickIdx / dimY;
        const auto y = stickIdx - x * dimY;
        for (SizeType z = 0; z < freqView.dim_inner(); ++z) {
          fullView(x, y, z) = std::complex<double>(dis(sharedRandGen), dis(sharedRandGen));
        }
      }
    }

    // copy data into sticks
    SizeType count = 0;
    for (const auto& stickIdx : paramPtr_->z_stick_xy_indices(comm_.rank())) {
      const auto x = stickIdx / dimY;
      const auto y = stickIdx - x * dimY;
      for (SizeType z = 0; z < freqView.dim_inner(); ++z) {
        freqView(count, z) = fullView(x, y, z);
      }
      ++count;
    }
  }

  MPICommunicatorHandle comm_;
  std::shared_ptr<Parameters> paramPtr_;
  HostArray<std::complex<double>> array1_;
  HostArray<std::complex<double>> array2_;
  HostArray<std::complex<double>> fullArray_;
};

static void check_space_domain(const HostArrayView3D<std::complex<double>>& realView,
                               const HostArrayView3D<std::complex<double>>& fullView,
                               const SizeType planeOffset, const SizeType numLocalXYPlanes) {
  for (SizeType z = 0; z < numLocalXYPlanes; ++z) {
    for (SizeType x = 0; x < fullView.dim_outer(); ++x) {
      for (SizeType y = 0; y < fullView.dim_mid(); ++y) {
        EXPECT_EQ(realView(z, x, y).real(), fullView(x, y, z + planeOffset).real());
        EXPECT_EQ(realView(z, x, y).imag(), fullView(x, y, z + planeOffset).imag());
      }
    }
  }
}

static void check_freq_domain(const HostArrayView2D<std::complex<double>>& freqView,
                              const HostArrayView3D<std::complex<double>>& fullView,
                              HostArrayConstView1D<int> xyIndices) {
  for (SizeType stickIdx = 0; stickIdx < freqView.dim_outer(); ++stickIdx) {
    const auto x = xyIndices(stickIdx) / fullView.dim_outer();
    const auto y = xyIndices(stickIdx) - x * fullView.dim_outer();
    for (SizeType z = 0; z < freqView.dim_inner(); ++z) {
      EXPECT_EQ(freqView(stickIdx, z).real(), fullView(x, y, z).real());
      EXPECT_EQ(freqView(stickIdx, z).imag(), fullView(x, y, z).imag());
    }
  }
}

TEST_F(TransposeTest, Unbuffered) {
  GTEST_MPI_GUARD
  auto freqXYView = create_3d_view(array2_, 0, paramPtr_->num_xy_planes(comm_.rank()),
                                   paramPtr_->dim_x(), paramPtr_->dim_y());
  auto freqView =
      create_2d_view(array1_, 0, paramPtr_->num_z_sticks(comm_.rank()), paramPtr_->dim_z());
  auto fullView =
      create_3d_view(fullArray_, 0, paramPtr_->dim_x(), paramPtr_->dim_y(), paramPtr_->dim_z());

  TransposeMPIUnbufferedHost<double> transpose(paramPtr_, comm_, freqXYView, freqView);

  transpose.backward();
  check_space_domain(freqXYView, fullView, paramPtr_->xy_plane_offset(comm_.rank()),
                     paramPtr_->num_xy_planes(comm_.rank()));

  transpose.forward();
  check_freq_domain(freqView, fullView, paramPtr_->z_stick_xy_indices(comm_.rank()));
}

TEST_F(TransposeTest, CompactBuffered) {
  GTEST_MPI_GUARD
  auto freqXYView = create_3d_view(array2_, 0, paramPtr_->num_xy_planes(comm_.rank()),
                                   paramPtr_->dim_x(), paramPtr_->dim_y());
  auto freqView =
      create_2d_view(array1_, 0, paramPtr_->num_z_sticks(comm_.rank()), paramPtr_->dim_z());
  auto fullView =
      create_3d_view(fullArray_, 0, paramPtr_->dim_x(), paramPtr_->dim_y(), paramPtr_->dim_z());

  auto transposeBufferZ = create_1d_view(
      array2_, 0, paramPtr_->total_num_xy_planes() * paramPtr_->num_z_sticks(comm_.rank()));
  auto transposeBufferXY = create_1d_view(
      array1_, 0, paramPtr_->total_num_z_sticks() * paramPtr_->num_xy_planes(comm_.rank()));

  TransposeMPICompactBufferedHost<double, double> transpose(paramPtr_, comm_, freqXYView, freqView,
                                                            transposeBufferXY, transposeBufferZ);

  transpose.backward();
  check_space_domain(freqXYView, fullView, paramPtr_->xy_plane_offset(comm_.rank()),
                     paramPtr_->num_xy_planes(comm_.rank()));
  transpose.forward();
  check_freq_domain(freqView, fullView, paramPtr_->z_stick_xy_indices(comm_.rank()));
}

TEST_F(TransposeTest, Buffered) {
  GTEST_MPI_GUARD
  auto freqXYView = create_3d_view(array2_, 0, paramPtr_->num_xy_planes(comm_.rank()),
                                   paramPtr_->dim_x(), paramPtr_->dim_y());
  auto freqView =
      create_2d_view(array1_, 0, paramPtr_->num_z_sticks(comm_.rank()), paramPtr_->dim_z());
  auto fullView =
      create_3d_view(fullArray_, 0, paramPtr_->dim_x(), paramPtr_->dim_y(), paramPtr_->dim_z());

  auto transposeBufferZ = create_1d_view(
      array2_, 0, paramPtr_->max_num_z_sticks() * paramPtr_->max_num_xy_planes() * comm_.size());
  auto transposeBufferXY = create_1d_view(
      array1_, 0, paramPtr_->max_num_z_sticks() * paramPtr_->max_num_xy_planes() * comm_.size());
  TransposeMPIBufferedHost<double, double> transpose(paramPtr_, comm_, freqXYView, freqView,
                                                     transposeBufferXY, transposeBufferZ);

  transpose.backward();
  check_space_domain(freqXYView, fullView, paramPtr_->xy_plane_offset(comm_.rank()),
                     paramPtr_->num_xy_planes(comm_.rank()));
  transpose.forward();
  check_freq_domain(freqView, fullView, paramPtr_->z_stick_xy_indices(comm_.rank()));
}
