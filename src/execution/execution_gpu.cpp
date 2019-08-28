/*
 * Copyright (c) 2019 ETH Zurich, Simon Frasch
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include "execution/execution_gpu.hpp"
#include "fft/transform_1d_gpu.hpp"
#include "fft/transform_2d_gpu.hpp"
#include "fft/transform_real_2d_gpu.hpp"
#include "gpu_util/gpu_pointer_translation.hpp"
#include "gpu_util/gpu_transfer.hpp"
#include "memory/array_view_utility.hpp"
#include "parameters/parameters.hpp"
#include "symmetry/symmetry_gpu.hpp"
#include "timing/timing.hpp"
#include "transpose/transpose_gpu.hpp"
#include "transpose/transpose_mpi_buffered_gpu.hpp"
#include "transpose/transpose_mpi_compact_buffered_gpu.hpp"
#include "transpose/transpose_mpi_unbuffered_gpu.hpp"

namespace spfft {

template <typename T>
ExecutionGPU<T>::ExecutionGPU(const int numThreads, std::shared_ptr<Parameters> param,
                              HostArray<std::complex<T>>& array1,
                              HostArray<std::complex<T>>& array2,
                              GPUArray<typename gpu::fft::ComplexType<T>::type>& gpuArray1,
                              GPUArray<typename gpu::fft::ComplexType<T>::type>& gpuArray2,
                              const std::shared_ptr<GPUArray<char>>& fftWorkBuffer)
    : stream_(false),
      numThreads_(numThreads),
      scalingFactor_(static_cast<T>(
          1.0 / static_cast<double>(param->dim_x() * param->dim_y() * param->dim_z()))),
      zStickSymmetry_(new Symmetry()),
      planeSymmetry_(new Symmetry()) {
  const SizeType numLocalZSticks = param->num_z_sticks(0);

  // frequency data with z-sticks
  freqDomainDataGPU_ = create_2d_view(gpuArray1, 0, numLocalZSticks, param->dim_z());
  freqDomainCompressedDataGPU_ =
      GPUArrayView1D<T>(reinterpret_cast<T*>(gpuArray2.data()),
                        param->local_value_indices().size() * 2, gpuArray2.device_id());

  // Z
  if (numLocalZSticks > 0) {
    transformZ_ = std::unique_ptr<TransformGPU>(
        new Transform1DGPU<T>(freqDomainDataGPU_, stream_, fftWorkBuffer));
    if (param->transform_type() == SPFFT_TRANS_R2C) {
      zStickSymmetry_.reset(new StickSymmetryGPU<T>(
          stream_, GPUArrayView1D<typename gpu::fft::ComplexType<T>::type>(
                       freqDomainDataGPU_.data() +
                           freqDomainDataGPU_.index(param->zero_zero_stick_index(), 0),
                       freqDomainDataGPU_.dim_inner(), freqDomainDataGPU_.device_id())));
    }
  }

  if (numLocalZSticks > 0 && param->local_value_indices().size() > 0) {
    compression_.reset(new CompressionGPU(param));
  }

  // Transpose
  auto freqDomainXYGPU = create_3d_view(gpuArray2, 0, param->dim_z(), param->dim_y(),
                                        param->dim_x_freq()); // must not overlap with z-sticks
  transpose_.reset(new TransposeGPU<T>(param, stream_, freqDomainXYGPU, freqDomainDataGPU_));

  // XY
  if (param->num_xy_planes(0) > 0) {
    if (param->transform_type() == SPFFT_TRANS_R2C) {
      planeSymmetry_.reset(new PlaneSymmetryGPU<T>(stream_, freqDomainXYGPU));
      // NOTE: param->dim_x() != param->dim_x_freq()
      spaceDomainDataExternalHost_ =
          create_new_type_3d_view<T>(array1, param->dim_z(), param->dim_y(), param->dim_x());
      spaceDomainDataExternalGPU_ =
          create_new_type_3d_view<T>(gpuArray1, param->dim_z(), param->dim_y(), param->dim_x());

      transformXY_ = std::unique_ptr<TransformGPU>(new TransformReal2DGPU<T>(
          spaceDomainDataExternalGPU_, freqDomainXYGPU, stream_, fftWorkBuffer));

    } else {
      spaceDomainDataExternalHost_ = create_new_type_3d_view<T>(
          array1, param->dim_z(), param->dim_y(), 2 * param->dim_x_freq());
      spaceDomainDataExternalGPU_ = create_new_type_3d_view<T>(
          freqDomainXYGPU, param->dim_z(), param->dim_y(), 2 * param->dim_x_freq());

      transformXY_ = std::unique_ptr<TransformGPU>(
          new Transform2DGPU<T>(freqDomainXYGPU, stream_, fftWorkBuffer));
    }
  }
}

#ifdef SPFFT_MPI
template <typename T>
ExecutionGPU<T>::ExecutionGPU(MPICommunicatorHandle comm, const SpfftExchangeType exchangeType,
                              const int numThreads, std::shared_ptr<Parameters> param,
                              HostArray<std::complex<T>>& array1,
                              HostArray<std::complex<T>>& array2,
                              GPUArray<typename gpu::fft::ComplexType<T>::type>& gpuArray1,
                              GPUArray<typename gpu::fft::ComplexType<T>::type>& gpuArray2,
                              const std::shared_ptr<GPUArray<char>>& fftWorkBuffer)
    : stream_(false),
      numThreads_(numThreads),
      scalingFactor_(static_cast<T>(
          1.0 / static_cast<double>(param->dim_x() * param->dim_y() * param->dim_z()))),
      zStickSymmetry_(new Symmetry()),
      planeSymmetry_(new Symmetry()) {
  assert(array1.data() != array2.data());
  assert(gpuArray1.data() != gpuArray2.data());
  assert(gpuArray1.device_id() == gpuArray2.device_id());

  const SizeType numLocalZSticks = param->num_z_sticks(comm.rank());
  const SizeType numLocalXYPlanes = param->num_xy_planes(comm.rank());

  freqDomainDataGPU_ = create_2d_view(gpuArray1, 0, numLocalZSticks, param->dim_z());
  freqDomainCompressedDataGPU_ =
      GPUArrayView1D<T>(reinterpret_cast<T*>(gpuArray2.data()),
                        param->local_value_indices().size() * 2, gpuArray2.device_id());

  auto freqDomainXYGPU = create_3d_view(gpuArray2, 0, numLocalXYPlanes, param->dim_y(),
                                        param->dim_x_freq()); // must not overlap with z-sticks

  // Z
  if (numLocalZSticks > 0) {
    transformZ_ = std::unique_ptr<TransformGPU>(
        new Transform1DGPU<T>(freqDomainDataGPU_, stream_, fftWorkBuffer));

    if (param->transform_type() == SPFFT_TRANS_R2C &&
        param->zero_zero_stick_index() < freqDomainDataGPU_.dim_outer()) {
      zStickSymmetry_.reset(new StickSymmetryGPU<T>(
          stream_, GPUArrayView1D<typename gpu::fft::ComplexType<T>::type>(
                       freqDomainDataGPU_.data() +
                           freqDomainDataGPU_.index(param->zero_zero_stick_index(), 0),
                       freqDomainDataGPU_.dim_inner(), freqDomainDataGPU_.device_id())));
    }
  }

  if (numLocalZSticks > 0) {
    compression_.reset(new CompressionGPU(param));
  }

  // XY
  if (numLocalXYPlanes > 0) {
    if (param->transform_type() == SPFFT_TRANS_R2C) {
      // NOTE: param->dim_x() != param->dim_x_freq()
      spaceDomainDataExternalHost_ =
          create_new_type_3d_view<T>(array1, numLocalXYPlanes, param->dim_y(), param->dim_x());
      spaceDomainDataExternalGPU_ =
          create_new_type_3d_view<T>(gpuArray1, numLocalXYPlanes, param->dim_y(), param->dim_x());

      transformXY_ = std::unique_ptr<TransformGPU>(new TransformReal2DGPU<T>(
          spaceDomainDataExternalGPU_, freqDomainXYGPU, stream_, fftWorkBuffer));

      planeSymmetry_.reset(new PlaneSymmetryGPU<T>(stream_, freqDomainXYGPU));

    } else {
      spaceDomainDataExternalHost_ = create_new_type_3d_view<T>(
          array1, numLocalXYPlanes, param->dim_y(), 2 * param->dim_x_freq());
      spaceDomainDataExternalGPU_ = create_new_type_3d_view<T>(
          freqDomainXYGPU, numLocalXYPlanes, param->dim_y(), 2 * param->dim_x_freq());

      transformXY_ = std::unique_ptr<TransformGPU>(
          new Transform2DGPU<T>(freqDomainXYGPU, stream_, fftWorkBuffer));
    }
  }

  switch (exchangeType) {
    case SpfftExchangeType::SPFFT_EXCH_UNBUFFERED: {
      auto freqDomainDataHost = create_2d_view(array1, 0, numLocalZSticks, param->dim_z());
      auto freqDomainXYHost =
          create_3d_view(array2, 0, numLocalXYPlanes, param->dim_y(), param->dim_x_freq());
      transpose_.reset(
          new TransposeMPIUnbufferedGPU<T>(param, comm, freqDomainXYHost, freqDomainXYGPU, stream_,
                                           freqDomainDataHost, freqDomainDataGPU_, stream_));
    } break;
    case SpfftExchangeType::SPFFT_EXCH_COMPACT_BUFFERED: {
      const auto bufferZSize = param->total_num_xy_planes() * param->num_z_sticks(comm.rank());
      const auto bufferXYSize = param->total_num_z_sticks() * param->num_xy_planes(comm.rank());
      auto transposeBufferZ = create_1d_view(array2, 0, bufferZSize);
      auto transposeBufferZGPU = create_1d_view(gpuArray2, 0, bufferZSize);
      auto transposeBufferXY = create_1d_view(array1, 0, bufferXYSize);
      auto transposeBufferXYGPU = create_1d_view(gpuArray1, 0, bufferXYSize);
      transpose_.reset(new TransposeMPICompactBufferedGPU<T, T>(
          param, comm, transposeBufferXY, freqDomainXYGPU, transposeBufferXYGPU, stream_,
          transposeBufferZ, freqDomainDataGPU_, transposeBufferZGPU, stream_));
    } break;
    case SpfftExchangeType::SPFFT_EXCH_COMPACT_BUFFERED_FLOAT: {
      const auto bufferZSize = param->total_num_xy_planes() * param->num_z_sticks(comm.rank());
      const auto bufferXYSize = param->total_num_z_sticks() * param->num_xy_planes(comm.rank());
      auto transposeBufferZ = create_1d_view(array2, 0, bufferZSize);
      auto transposeBufferZGPU = create_1d_view(gpuArray2, 0, bufferZSize);
      auto transposeBufferXY = create_1d_view(array1, 0, bufferXYSize);
      auto transposeBufferXYGPU = create_1d_view(gpuArray1, 0, bufferXYSize);
      transpose_.reset(new TransposeMPICompactBufferedGPU<T, float>(
          param, comm, transposeBufferXY, freqDomainXYGPU, transposeBufferXYGPU, stream_,
          transposeBufferZ, freqDomainDataGPU_, transposeBufferZGPU, stream_));
    } break;
    case SpfftExchangeType::SPFFT_EXCH_BUFFERED: {
      const auto bufferSize = param->max_num_z_sticks() * param->max_num_xy_planes() * comm.size();
      auto transposeBufferZ = create_1d_view(array2, 0, bufferSize);
      auto transposeBufferZGPU = create_1d_view(gpuArray2, 0, bufferSize);
      auto transposeBufferXY = create_1d_view(array1, 0, bufferSize);
      auto transposeBufferXYGPU = create_1d_view(gpuArray1, 0, bufferSize);
      transpose_.reset(new TransposeMPIBufferedGPU<T, T>(
          param, comm, transposeBufferXY, freqDomainXYGPU, transposeBufferXYGPU, stream_,
          transposeBufferZ, freqDomainDataGPU_, transposeBufferZGPU, stream_));
    } break;
    case SpfftExchangeType::SPFFT_EXCH_BUFFERED_FLOAT: {
      const auto bufferSize = param->max_num_z_sticks() * param->max_num_xy_planes() * comm.size();
      auto transposeBufferZ = create_1d_view(array2, 0, bufferSize);
      auto transposeBufferZGPU = create_1d_view(gpuArray2, 0, bufferSize);
      auto transposeBufferXY = create_1d_view(array1, 0, bufferSize);
      auto transposeBufferXYGPU = create_1d_view(gpuArray1, 0, bufferSize);
      transpose_.reset(new TransposeMPIBufferedGPU<T, float>(
          param, comm, transposeBufferXY, freqDomainXYGPU, transposeBufferXYGPU, stream_,
          transposeBufferZ, freqDomainDataGPU_, transposeBufferZGPU, stream_));
    } break;
    default:
      throw InvalidParameterError();
  }
}

// instatiate templates for float and double
#endif

template <typename T>
auto ExecutionGPU<T>::forward_xy(const SpfftProcessingUnitType inputLocation) -> void {

  // Check for any preceding errors before starting execution
  if (gpu::get_last_error() != gpu::status::Success) {
    throw GPUPrecedingError();
  }

  // XY
  if (transformXY_) {
    if (inputLocation == SpfftProcessingUnitType::SPFFT_PU_HOST) {
      copy_to_gpu_async(stream_, spaceDomainDataExternalHost_, spaceDomainDataExternalGPU_);
    }
    transformXY_->forward();
  }

  // transpose
  if (transformXY_) transpose_->pack_forward();
}

template <typename T>
auto ExecutionGPU<T>::forward_exchange(const bool nonBlockingExchange) -> void {
  HOST_TIMING_SCOPED("exchange_start")
  transpose_->exchange_forward_start(nonBlockingExchange);
}

template <typename T>
auto ExecutionGPU<T>::forward_z(T* output, const SpfftScalingType scalingType) -> void {

  HOST_TIMING_START("exechange_fininalize");
  transpose_->exchange_forward_finalize();
  HOST_TIMING_STOP("exechange_fininalize");

  if (transformZ_) transpose_->unpack_forward();

  // Z
  if (transformZ_) transformZ_->forward();

  // Compress
  if (compression_) {
    T* outputPtrHost = nullptr;
    T* outputPtrGPU = nullptr;
    std::tie(outputPtrHost, outputPtrGPU) = translate_gpu_pointer(output);

    if (outputPtrGPU == nullptr) {
      // output on HOST
      compression_->compress(stream_, freqDomainDataGPU_, freqDomainCompressedDataGPU_.data(),
                             scalingType == SpfftScalingType::SPFFT_FULL_SCALING, scalingFactor_);

      gpu::check_status(
          gpu::memcpy_async(static_cast<void*>(outputPtrHost),
                            static_cast<const void*>(freqDomainCompressedDataGPU_.data()),
                            freqDomainCompressedDataGPU_.size() *
                                sizeof(decltype(*(freqDomainCompressedDataGPU_.data()))),
                            gpu::flag::MemcpyDeviceToHost, stream_.get()));
    } else {
      // output on GPU
      compression_->compress(stream_, freqDomainDataGPU_, outputPtrGPU,
                             scalingType == SpfftScalingType::SPFFT_FULL_SCALING, scalingFactor_);
    }
  }
}

template <typename T>
auto ExecutionGPU<T>::backward_z(const T* input) -> void {

  // Check for any preceding errors before starting execution
  if (gpu::get_last_error() != gpu::status::Success) {
    throw GPUPrecedingError();
  }

  // decompress
  if (compression_) {
    const T* inputPtrHost = nullptr;
    const T* inputPtrGPU = nullptr;
    std::tie(inputPtrHost, inputPtrGPU) = translate_gpu_pointer(input);

    if (inputPtrGPU == nullptr) {
      // input on HOST
      gpu::check_status(
          gpu::memcpy_async(static_cast<void*>(freqDomainCompressedDataGPU_.data()),
                            static_cast<const void*>(inputPtrHost),
                            freqDomainCompressedDataGPU_.size() *
                                sizeof(decltype(*(freqDomainCompressedDataGPU_.data()))),
                            gpu::flag::MemcpyHostToDevice, stream_.get()));
      compression_->decompress(stream_, freqDomainCompressedDataGPU_.data(), freqDomainDataGPU_);
    } else {
      // input on GPU
      compression_->decompress(stream_, inputPtrGPU, freqDomainDataGPU_);
    }
  }

  // Z
  if (transformZ_) {
    zStickSymmetry_->apply();
    transformZ_->backward();
  }

  // transpose
  if (transformZ_) transpose_->pack_backward();
}
template <typename T>
auto ExecutionGPU<T>::backward_exchange(const bool nonBlockingExchange) -> void {
  transpose_->exchange_backward_start(nonBlockingExchange);
}

template <typename T>
auto ExecutionGPU<T>::backward_xy(const SpfftProcessingUnitType outputLocation) -> void {

  HOST_TIMING_START("exechange_fininalize");
  transpose_->exchange_backward_finalize();
  HOST_TIMING_STOP("exechange_fininalize");

  if (transformXY_) transpose_->unpack_backward();

  // XY
  if (transformXY_) {
    planeSymmetry_->apply();
    transformXY_->backward();
    if (outputLocation & SpfftProcessingUnitType::SPFFT_PU_HOST) {
      copy_from_gpu_async(stream_, spaceDomainDataExternalGPU_, spaceDomainDataExternalHost_);
    }
  }
}

template <typename T>
auto ExecutionGPU<T>::synchronize() -> void {
  gpu::stream_synchronize(stream_.get());
}

template <typename T>
auto ExecutionGPU<T>::space_domain_data_host() -> HostArrayView3D<T> {
  return spaceDomainDataExternalHost_;
}

template <typename T>
auto ExecutionGPU<T>::space_domain_data_gpu() -> GPUArrayView3D<T> {
  return spaceDomainDataExternalGPU_;
}

// instatiate templates for float and double
template class ExecutionGPU<double>;
#ifdef SPFFT_SINGLE_PRECISION
template class ExecutionGPU<float>;
#endif

} // namespace spfft
