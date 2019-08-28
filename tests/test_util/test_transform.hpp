#ifndef SPFFT_TEST_TRANSFORM_HPP
#define SPFFT_TEST_TRANSFORM_HPP

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
#include "parameters/parameters.hpp"
#include "spfft/grid.hpp"
#include "spfft/transform.hpp"
#include "test_util/test_check_values.hpp"
#include "test_util/generate_indices.hpp"
#include "util/common_types.hpp"

#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
#include "gpu_util/gpu_fft_api.hpp"
#include "gpu_util/gpu_transfer.hpp"
#include "memory/gpu_array.hpp"
#endif

using namespace spfft;
class TransformTest
    : public ::testing::TestWithParam<
          std::tuple<SpfftExchangeType, SpfftProcessingUnitType, int, int, int, bool>> {
protected:
  TransformTest()
      : dimX_(std::get<2>(GetParam())),
        dimY_(std::get<3>(GetParam())),
        dimZ_(std::get<4>(GetParam())),
        fftwArray_(dimX_ * dimY_ * dimZ_),
        fftwView_(create_3d_view(fftwArray_, 0, dimX_, dimY_, dimZ_)),
        centeredIndices_(std::get<5>(GetParam())) {
    // initialize ffw plans
    fftwPlanBackward_ =
        fftw_plan_dft_3d(dimX_, dimY_, dimZ_, (fftw_complex*)fftwArray_.data(),
                         (fftw_complex*)fftwArray_.data(), FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwPlanForward_ =
        fftw_plan_dft_3d(dimX_, dimY_, dimZ_, (fftw_complex*)fftwArray_.data(),
                         (fftw_complex*)fftwArray_.data(), FFTW_FORWARD, FFTW_ESTIMATE);
  }

  inline auto test_backward_c2c(const std::vector<double>& zStickDistribution,
                                const std::vector<double>& xyPlaneDistribution) -> void;

  inline auto test_forward_c2c(const std::vector<double>& zStickDistribution,
                               const std::vector<double>& xyPlaneDistribution) -> void;

  inline auto test_r2c(const std::vector<double>& xyPlaneDistribution) -> void;

  virtual auto comm_rank() -> SizeType { return 0; }

  virtual auto comm_size() -> SizeType { return 1; }

  virtual auto grid() -> Grid& = 0;

  ~TransformTest() override {
    if (fftwPlanBackward_) fftw_destroy_plan(fftwPlanBackward_);
    if (fftwPlanForward_) fftw_destroy_plan(fftwPlanForward_);
    fftwPlanBackward_ = nullptr;
    fftwPlanForward_ = nullptr;
  }

  int dimX_, dimY_, dimZ_;
  HostArray<std::complex<double>> fftwArray_;
  HostArrayView3D<std::complex<double>> fftwView_;
  fftw_plan fftwPlanBackward_ = nullptr;
  fftw_plan fftwPlanForward_ = nullptr;
  bool centeredIndices_;
};

auto TransformTest::test_backward_c2c(const std::vector<double>& zStickDistribution,
                                      const std::vector<double>& xyPlaneDistribution) -> void {
  std::mt19937 randGen(42);
  std::uniform_real_distribution<double> uniformRandDis(0.0, 1.0);
  auto valueIndicesPerRank =
      create_value_indices(randGen, zStickDistribution, 0.7, 0.7, dimX_, dimY_, dimZ_, false);
  const int numLocalXYPlanes =
      calculate_num_local_xy_planes(comm_rank(), dimZ_, xyPlaneDistribution);

  // assign values to fftw input
  for (const auto& valueIndices : valueIndicesPerRank) {
    for (std::size_t i = 0; i < valueIndices.size(); i += 3) {
      fftwView_(valueIndices[i], valueIndices[i + 1], valueIndices[i + 2]) =
          std::complex<double>(uniformRandDis(randGen), uniformRandDis(randGen));
    }
  }

  // extract local rank values
  std::vector<std::complex<double>> values(valueIndicesPerRank[comm_rank()].size() / 3);
  for (std::size_t i = 0; i < values.size(); ++i) {
    const auto x = valueIndicesPerRank[comm_rank()][i * 3];
    const auto y = valueIndicesPerRank[comm_rank()][i * 3 + 1];
    const auto z = valueIndicesPerRank[comm_rank()][i * 3 + 2];
    values[i] = fftwView_(x, y, z);
  }

  if (centeredIndices_) {
    center_indices(dimX_, dimY_, dimZ_, valueIndicesPerRank);
  }

  auto transform = grid().create_transform(
      std::get<1>(GetParam()), SpfftTransformType::SPFFT_TRANS_C2C, dimX_, dimY_, dimZ_,
      numLocalXYPlanes, values.size(), SpfftIndexFormatType::SPFFT_INDEX_TRIPLETS,
      valueIndicesPerRank[comm_rank()].data());

  HostArrayView3D<std::complex<double>> realView(
      reinterpret_cast<std::complex<double>*>(
          transform.space_domain_data(SpfftProcessingUnitType::SPFFT_PU_HOST)),
      numLocalXYPlanes, dimY_, dimX_, false);

  fftw_execute(fftwPlanBackward_);

#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
  if (std::get<1>(GetParam()) == SpfftProcessingUnitType::SPFFT_PU_GPU) {
    // copy frequency values to GPU
    GPUArray<typename gpu::fft::ComplexType<double>::type> valuesGPU(values.size());
    copy_to_gpu(values, valuesGPU);

    // transform
    transform.backward(reinterpret_cast<double*>(valuesGPU.data()),
                       SpfftProcessingUnitType::SPFFT_PU_GPU);
    // run twice to ensure memory is zeroed correctly
    transform.backward(reinterpret_cast<double*>(valuesGPU.data()),
                       SpfftProcessingUnitType::SPFFT_PU_GPU);

    // use transform buffer to copy values
    GPUArrayView3D<typename gpu::fft::ComplexType<double>::type> realViewGPU(
        reinterpret_cast<typename gpu::fft::ComplexType<double>::type*>(
            transform.space_domain_data(SpfftProcessingUnitType::SPFFT_PU_GPU)),
        numLocalXYPlanes, dimY_, dimX_, false);
    copy_from_gpu(realViewGPU, realView);
  }
#endif
  if (std::get<1>(GetParam()) == SpfftProcessingUnitType::SPFFT_PU_HOST) {
    transform.backward(reinterpret_cast<double*>(values.data()),
                       SpfftProcessingUnitType::SPFFT_PU_HOST);
    // run twice to ensure memory is zeroed correctly
    transform.backward(reinterpret_cast<double*>(values.data()),
                       SpfftProcessingUnitType::SPFFT_PU_HOST);
  }
  check_c2c_space_domain(realView, fftwView_, transform.local_z_offset(), numLocalXYPlanes);
}

auto TransformTest::test_forward_c2c(const std::vector<double>& zStickDistribution,
                                     const std::vector<double>& xyPlaneDistribution) -> void {
  std::mt19937 randGen(42);
  std::uniform_real_distribution<double> uniformRandDis(0.0, 1.0);
  auto valueIndicesPerRank =
      create_value_indices(randGen, zStickDistribution, 0.7, 0.7, dimX_, dimY_, dimZ_, false);
  const int numLocalXYPlanes =
      calculate_num_local_xy_planes(comm_rank(), dimZ_, xyPlaneDistribution);

  // assign values to fftw input
  for (const auto& valueIndices : valueIndicesPerRank) {
    for (std::size_t i = 0; i < valueIndices.size(); i += 3) {
      fftwView_(valueIndices[i], valueIndices[i + 1], valueIndices[i + 2]) =
          std::complex<double>(uniformRandDis(randGen), uniformRandDis(randGen));
    }
  }

  std::vector<std::complex<double>> freqValues(valueIndicesPerRank[comm_rank()].size() / 3);

  if (centeredIndices_) {
    center_indices(dimX_, dimY_, dimZ_, valueIndicesPerRank);
  }

  auto transform = grid().create_transform(
      std::get<1>(GetParam()), SpfftTransformType::SPFFT_TRANS_C2C, dimX_, dimY_, dimZ_,
      numLocalXYPlanes, freqValues.size(), SpfftIndexFormatType::SPFFT_INDEX_TRIPLETS,
      valueIndicesPerRank[comm_rank()].data());

  HostArrayView3D<std::complex<double>> realView(
      reinterpret_cast<std::complex<double>*>(
          transform.space_domain_data(SpfftProcessingUnitType::SPFFT_PU_HOST)),
      numLocalXYPlanes, dimY_, dimX_, false);

  fftw_execute(fftwPlanBackward_);

  // copy space domain values from fftw buffer
  const auto zOffset = transform.local_z_offset();
  for (int z = 0; z < numLocalXYPlanes; ++z) {
    for (int y = 0; y < dimY_; ++y) {
      for (int x = 0; x < dimX_; ++x) {
        realView(z, y, x) = fftwView_(x, y, z + zOffset);
      }
    }
  }

  fftw_execute(fftwPlanForward_);

#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
  if (std::get<1>(GetParam()) == SpfftProcessingUnitType::SPFFT_PU_GPU) {
    // use transform buffer to copy values
    GPUArrayView3D<typename gpu::fft::ComplexType<double>::type> realViewGPU(
        reinterpret_cast<typename gpu::fft::ComplexType<double>::type*>(
            transform.space_domain_data(SpfftProcessingUnitType::SPFFT_PU_GPU)),
        numLocalXYPlanes, dimY_, dimX_, false);
    copy_to_gpu(realView, realViewGPU);

    GPUArray<typename gpu::fft::ComplexType<double>::type> freqValuesGPU(freqValues.size());
    transform.forward(SpfftProcessingUnitType::SPFFT_PU_GPU,
                      reinterpret_cast<double*>(freqValuesGPU.data()));
    copy_from_gpu(freqValuesGPU, freqValues);
  }
#endif
  if (std::get<1>(GetParam()) == SpfftProcessingUnitType::SPFFT_PU_HOST) {
    transform.forward(SpfftProcessingUnitType::SPFFT_PU_HOST,
                      reinterpret_cast<double*>(freqValues.data()));
  }

  check_freq_domain(freqValues, fftwView_, valueIndicesPerRank[comm_rank()]);
}

auto TransformTest::test_r2c(const std::vector<double>& xyPlaneDistribution) -> void {
  std::mt19937 randGen(42);
  std::uniform_real_distribution<double> uniformRandDis(0.0, 1.0);

  // create full set of global z-sticks (up to dimX_ / 2 + 1, due to symmetry)
  std::vector<double> zStickDistribution(xyPlaneDistribution.size(), 1.0);
  auto valueIndicesPerRank =
      create_value_indices(randGen, zStickDistribution, 1.0, 1.0, dimX_, dimY_, dimZ_, true);
  const int numLocalXYPlanes =
      calculate_num_local_xy_planes(comm_rank(), dimZ_, xyPlaneDistribution);

  // assign values to fftw input
  for (const auto& valueIndices : valueIndicesPerRank) {
    for (std::size_t i = 0; i < valueIndices.size(); i += 3) {
      fftwView_(valueIndices[i], valueIndices[i + 1], valueIndices[i + 2]) =
          std::complex<double>(uniformRandDis(randGen), 0.0);
    }
  }

  std::vector<std::complex<double>> freqValues(valueIndicesPerRank[comm_rank()].size() / 3);

  if (centeredIndices_) {
    center_indices(dimX_, dimY_, dimZ_, valueIndicesPerRank);
  }

  auto transform = grid().create_transform(
      std::get<1>(GetParam()), SpfftTransformType::SPFFT_TRANS_R2C, dimX_, dimY_, dimZ_,
      numLocalXYPlanes, freqValues.size(), SpfftIndexFormatType::SPFFT_INDEX_TRIPLETS,
      valueIndicesPerRank[comm_rank()].data());

  HostArrayView3D<double> realView(
      transform.space_domain_data(SpfftProcessingUnitType::SPFFT_PU_HOST), numLocalXYPlanes, dimY_,
      dimX_, false);

  // copy space domain values from fftw buffer
  const auto zOffset = transform.local_z_offset();
  for (int z = 0; z < numLocalXYPlanes; ++z) {
    for (int y = 0; y < dimY_; ++y) {
      for (int x = 0; x < dimX_; ++x) {
        realView(z, y, x) = fftwView_(x, y, z + zOffset).real();
      }
    }
  }

  // check forward
  transform.forward(SpfftProcessingUnitType::SPFFT_PU_HOST,
                    reinterpret_cast<double*>(freqValues.data()));
  fftw_execute(fftwPlanForward_);
  check_freq_domain(freqValues, fftwView_, valueIndicesPerRank[comm_rank()]);

  // check backward
  transform.backward(reinterpret_cast<double*>(freqValues.data()),
                     SpfftProcessingUnitType::SPFFT_PU_HOST);
  fftw_execute(fftwPlanBackward_);
  check_r2c_space_domain(realView, fftwView_, transform.local_z_offset(), numLocalXYPlanes);
}

#endif
