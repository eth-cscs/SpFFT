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
#include "util/common_types.hpp"

#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
#include "execution/execution_gpu.hpp"
#include "memory/gpu_array.hpp"
#include "transpose/transpose_mpi_buffered_gpu.hpp"
#include "transpose/transpose_mpi_compact_buffered_gpu.hpp"
#include "transpose/transpose_mpi_unbuffered_gpu.hpp"

using namespace spfft;

class TransposeGPUTest : public ::testing::Test {
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
    gpuArray1_ = GPUArray<typename gpu::fft::ComplexType<double>::type>(array1_.size());
    gpuArray2_ = GPUArray<typename gpu::fft::ComplexType<double>::type>(array1_.size());

    // pinn arrays
    array1_.pin_memory();
    array2_.pin_memory();

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
        if (dis(sharedRandGen) < 0.5 && rankSelector(sharedRandGen) == comm_.size()) {
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
  GPUArray<typename gpu::fft::ComplexType<double>::type> gpuArray1_;
  GPUArray<typename gpu::fft::ComplexType<double>::type> gpuArray2_;
};

static void check_space_domain(const HostArrayView3D<std::complex<double>>& realView,
                               const HostArrayView3D<std::complex<double>>& fullView,
                               const SizeType planeOffset, const SizeType numLocalXYPlanes) {
  for (SizeType z = 0; z < numLocalXYPlanes; ++z) {
    for (SizeType x = 0; x < fullView.dim_outer(); ++x) {
      for (SizeType y = 0; y < fullView.dim_mid(); ++y) {
        EXPECT_EQ(realView(z, y, x).real(), fullView(x, y, z + planeOffset).real());
        EXPECT_EQ(realView(z, y, x).imag(), fullView(x, y, z + planeOffset).imag());
      }
    }
  }
}

static void check_freq_domain(const HostArrayView2D<std::complex<double>>& freqView,
                              const HostArrayView3D<std::complex<double>>& fullView,
                              HostArrayConstView1D<int> xyIndices) {
  for (SizeType stickIdx = 0; stickIdx < freqView.dim_outer(); ++stickIdx) {
    const auto x = stickIdx / fullView.dim_outer();
    const auto y = stickIdx - x * fullView.dim_outer();
    for (SizeType z = 0; z < freqView.dim_inner(); ++z) {
      EXPECT_EQ(freqView(stickIdx, z).real(), fullView(x, y, z).real());
      EXPECT_EQ(freqView(stickIdx, z).imag(), fullView(x, y, z).imag());
    }
  }
}

TEST_F(TransposeGPUTest, Buffered) {
  GTEST_MPI_GUARD
  auto freqXYView = create_3d_view(array2_, 0, paramPtr_->num_xy_planes(comm_.rank()),
                                   paramPtr_->dim_y(), paramPtr_->dim_x());
  auto freqXYViewGPU = create_3d_view(gpuArray2_, 0, paramPtr_->num_xy_planes(comm_.rank()),
                                      paramPtr_->dim_y(), paramPtr_->dim_x());
  auto freqView =
      create_2d_view(array1_, 0, paramPtr_->num_z_sticks(comm_.rank()), paramPtr_->dim_z());
  auto freqViewGPU =
      create_2d_view(gpuArray1_, 0, paramPtr_->num_z_sticks(comm_.rank()), paramPtr_->dim_z());

  auto fullView =
      create_3d_view(fullArray_, 0, paramPtr_->dim_x(), paramPtr_->dim_y(), paramPtr_->dim_z());

  GPUStreamHandle stream(false);
  auto transposeBufferZ = create_1d_view(
      array2_, 0, comm_.size() * paramPtr_->max_num_xy_planes() * paramPtr_->max_num_z_sticks());
  auto transposeBufferZGPU = create_1d_view(
      gpuArray2_, 0, comm_.size() * paramPtr_->max_num_xy_planes() * paramPtr_->max_num_z_sticks());
  auto transposeBufferXY = create_1d_view(
      array1_, 0, comm_.size() * paramPtr_->max_num_xy_planes() * paramPtr_->max_num_z_sticks());
  auto transposeBufferXYGPU = create_1d_view(
      gpuArray1_, 0, comm_.size() * paramPtr_->max_num_xy_planes() * paramPtr_->max_num_z_sticks());

  TransposeMPIBufferedGPU<double, double> transpose(
      paramPtr_, comm_, transposeBufferXY, freqXYViewGPU, transposeBufferXYGPU, stream,
      transposeBufferZ, freqViewGPU, transposeBufferZGPU, stream);

  copy_to_gpu_async(stream, freqView, freqViewGPU);
  transpose.backward();
  copy_from_gpu_async(stream, freqXYViewGPU, freqXYView);
  gpu::check_status(gpu::stream_synchronize(stream.get()));
  check_space_domain(freqXYView, fullView, paramPtr_->xy_plane_offset(comm_.rank()),
                     paramPtr_->num_xy_planes(comm_.rank()));

  transpose.forward();
  copy_from_gpu_async(stream, freqViewGPU, freqView);
  gpu::check_status(gpu::stream_synchronize(stream.get()));
  check_freq_domain(freqView, fullView, paramPtr_->z_stick_xy_indices(comm_.rank()));
}

TEST_F(TransposeGPUTest, CompactBuffered) {
  GTEST_MPI_GUARD
  auto freqXYView = create_3d_view(array2_, 0, paramPtr_->num_xy_planes(comm_.rank()),
                                   paramPtr_->dim_y(), paramPtr_->dim_x());
  auto freqXYViewGPU = create_3d_view(gpuArray2_, 0, paramPtr_->num_xy_planes(comm_.rank()),
                                      paramPtr_->dim_y(), paramPtr_->dim_x());
  auto freqView =
      create_2d_view(array1_, 0, paramPtr_->num_z_sticks(comm_.rank()), paramPtr_->dim_z());
  auto freqViewGPU =
      create_2d_view(gpuArray1_, 0, paramPtr_->num_z_sticks(comm_.rank()), paramPtr_->dim_z());

  auto fullView =
      create_3d_view(fullArray_, 0, paramPtr_->dim_x(), paramPtr_->dim_y(), paramPtr_->dim_z());

  GPUStreamHandle stream(false);
  auto transposeBufferZ = create_1d_view(
      array2_, 0, comm_.size() * paramPtr_->max_num_xy_planes() * paramPtr_->max_num_z_sticks());
  auto transposeBufferZGPU = create_1d_view(
      gpuArray2_, 0, comm_.size() * paramPtr_->max_num_xy_planes() * paramPtr_->max_num_z_sticks());
  auto transposeBufferXY = create_1d_view(
      array1_, 0, comm_.size() * paramPtr_->max_num_xy_planes() * paramPtr_->max_num_z_sticks());
  auto transposeBufferXYGPU = create_1d_view(
      gpuArray1_, 0, comm_.size() * paramPtr_->max_num_xy_planes() * paramPtr_->max_num_z_sticks());

  TransposeMPICompactBufferedGPU<double, double> transpose(
      paramPtr_, comm_, transposeBufferXY, freqXYViewGPU, transposeBufferXYGPU, stream,
      transposeBufferZ, freqViewGPU, transposeBufferZGPU, stream);

  copy_to_gpu_async(stream, freqView, freqViewGPU);
  transpose.pack_backward();
  transpose.backward();
  transpose.unpack_backward();
  copy_from_gpu_async(stream, freqXYViewGPU, freqXYView);
  gpu::check_status(gpu::stream_synchronize(stream.get()));
  check_space_domain(freqXYView, fullView, paramPtr_->xy_plane_offset(comm_.rank()),
                     paramPtr_->num_xy_planes(comm_.rank()));

  transpose.forward();
  copy_from_gpu_async(stream, freqViewGPU, freqView);
  gpu::check_status(gpu::stream_synchronize(stream.get()));
  check_freq_domain(freqView, fullView, paramPtr_->z_stick_xy_indices(comm_.rank()));
}

TEST_F(TransposeGPUTest, Unbuffered) {
  GTEST_MPI_GUARD
  auto freqXYView = create_3d_view(array2_, 0, paramPtr_->num_xy_planes(comm_.rank()),
                                   paramPtr_->dim_y(), paramPtr_->dim_x());
  auto freqXYViewGPU = create_3d_view(gpuArray2_, 0, paramPtr_->num_xy_planes(comm_.rank()),
                                      paramPtr_->dim_y(), paramPtr_->dim_x());
  auto freqView =
      create_2d_view(array1_, 0, paramPtr_->num_z_sticks(comm_.rank()), paramPtr_->dim_z());
  auto freqViewGPU =
      create_2d_view(gpuArray1_, 0, paramPtr_->num_z_sticks(comm_.rank()), paramPtr_->dim_z());

  auto fullView =
      create_3d_view(fullArray_, 0, paramPtr_->dim_x(), paramPtr_->dim_y(), paramPtr_->dim_z());

  GPUStreamHandle stream(false);

  TransposeMPIUnbufferedGPU<double> transpose(paramPtr_, comm_, freqXYView, freqXYViewGPU, stream,
                                              freqView, freqViewGPU, stream);

  copy_to_gpu_async(stream, freqView, freqViewGPU);
  transpose.backward();
  copy_from_gpu_async(stream, freqXYViewGPU, freqXYView);
  gpu::check_status(gpu::stream_synchronize(stream.get()));
  check_space_domain(freqXYView, fullView, paramPtr_->xy_plane_offset(comm_.rank()),
                     paramPtr_->num_xy_planes(comm_.rank()));

  transpose.forward();
  copy_from_gpu_async(stream, freqViewGPU, freqView);
  gpu::check_status(gpu::stream_synchronize(stream.get()));
  check_freq_domain(freqView, fullView, paramPtr_->z_stick_xy_indices(comm_.rank()));
}
#endif
