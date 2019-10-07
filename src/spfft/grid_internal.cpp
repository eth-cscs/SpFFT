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
#include "spfft/config.h"

#include <complex>
#include <memory>
#include "spfft/grid_internal.hpp"

#ifdef SPFFT_MPI
#include "mpi_util/mpi_check_status.hpp"
#include "mpi_util/mpi_datatype_handle.hpp"
#include "mpi_util/mpi_match_elementary_type.hpp"
#endif

#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
#include "gpu_util/gpu_device_guard.hpp"
#include "gpu_util/gpu_runtime_api.hpp"
#endif

namespace spfft {

template <typename T>
GridInternal<T>::GridInternal(int maxDimX, int maxDimY, int maxDimZ, int maxNumLocalZSticks,
                              SpfftProcessingUnitType executionUnit, int numThreads)
    : isLocal_(true),
      executionUnit_(executionUnit),
      deviceId_(0),
      numThreads_(numThreads),
      maxDimX_(maxDimX),
      maxDimY_(maxDimY),
      maxDimZ_(maxDimZ),
      maxNumLocalZSticks_(maxNumLocalZSticks),
      maxNumLocalXYPlanes_(maxDimZ) {
  // input check
  if (maxDimX <= 0 || maxDimY <= 0 || maxDimZ <= 0 || maxNumLocalZSticks < 0) {
    throw InvalidParameterError();
  }
  if (!(executionUnit &
        (SpfftProcessingUnitType::SPFFT_PU_HOST | SpfftProcessingUnitType::SPFFT_PU_GPU))) {
    throw InvalidParameterError();
  }

  // set number of threads to default omp value if not valid
  if (numThreads < 1) {
    numThreads = omp_get_max_threads();
    numThreads_ = omp_get_max_threads();
  }

  // allocate memory
  if (executionUnit & SpfftProcessingUnitType::SPFFT_PU_HOST) {
    arrayHost1_ = HostArray<ComplexType>(static_cast<SizeType>(maxDimX * maxDimY * maxDimZ));
    arrayHost2_ = HostArray<ComplexType>(static_cast<SizeType>(maxDimX * maxDimY * maxDimZ));
  }
  if (executionUnit & SpfftProcessingUnitType::SPFFT_PU_GPU) {
#if (defined(SPFFT_CUDA) || defined(SPFFT_ROCM))
    // store device id
    gpu::check_status(gpu::get_device(&deviceId_));

    if (arrayHost1_.empty()) {
      // not already created for CPU, which always requires at least as much memory
      arrayHost1_ = HostArray<ComplexType>(static_cast<SizeType>(maxNumLocalZSticks * maxDimZ));
      arrayHost2_ = HostArray<ComplexType>(static_cast<SizeType>(maxDimX * maxDimY * maxDimZ));
    }
    arrayHost1_.pin_memory();
    arrayHost2_.pin_memory();
    arrayGPU1_ = GPUArray<typename gpu::fft::ComplexType<ValueType>::type>(
        static_cast<SizeType>(maxNumLocalZSticks * maxDimZ));
    arrayGPU2_ = GPUArray<typename gpu::fft::ComplexType<ValueType>::type>(
        static_cast<SizeType>(maxDimX * maxDimY * maxDimZ));

    // each transform will resize the work buffer as needed
    fftWorkBuffer_.reset(new GPUArray<char>());

#else
    throw GPUSupportError();
#endif
  }
}

#ifdef SPFFT_MPI
template <typename T>
GridInternal<T>::GridInternal(int maxDimX, int maxDimY, int maxDimZ, int maxNumLocalZSticks,
                              int maxNumLocalXYPlanes, SpfftProcessingUnitType executionUnit,
                              int numThreads, MPI_Comm comm, SpfftExchangeType exchangeType)
    : isLocal_(false),
      executionUnit_(executionUnit),
      deviceId_(0),
      numThreads_(numThreads),
      maxDimX_(maxDimX),
      maxDimY_(maxDimY),
      maxDimZ_(maxDimZ),
      maxNumLocalZSticks_(maxNumLocalZSticks),
      maxNumLocalXYPlanes_(maxNumLocalXYPlanes),
      comm_(comm),
      exchangeType_(exchangeType) {
  // input check
  if (static_cast<long long int>(maxDimX) * static_cast<long long int>(maxDimY) *
          static_cast<long long int>(maxNumLocalXYPlanes) >
      std::numeric_limits<int>::max()) {
    throw OverflowError();
  }
  if (static_cast<long long int>(maxNumLocalZSticks) * static_cast<long long int>(maxDimZ) >
      std::numeric_limits<int>::max()) {
    throw OverflowError();
  }
  if (maxDimX <= 0 || maxDimY <= 0 || maxDimZ <= 0 || maxNumLocalZSticks < 0) {
    throw InvalidParameterError();
  }
  if (!(executionUnit &
        (SpfftProcessingUnitType::SPFFT_PU_HOST | SpfftProcessingUnitType::SPFFT_PU_GPU))) {
    throw InvalidParameterError();
  }
  if (exchangeType != SpfftExchangeType::SPFFT_EXCH_DEFAULT &&
      exchangeType != SpfftExchangeType::SPFFT_EXCH_BUFFERED &&
      exchangeType != SpfftExchangeType::SPFFT_EXCH_BUFFERED_FLOAT &&
      exchangeType != SpfftExchangeType::SPFFT_EXCH_COMPACT_BUFFERED &&
      exchangeType != SpfftExchangeType::SPFFT_EXCH_COMPACT_BUFFERED_FLOAT &&
      exchangeType != SpfftExchangeType::SPFFT_EXCH_UNBUFFERED) {
    throw InvalidParameterError();
  }

  // compare parameters between ranks
  {
    int errorDetected = 0;
    int exchangeAll = exchangeType;
    int executionUnitAll = executionUnit;

    // Bitwise or will lead to a mismatch on at least one rank if not all values are equal
    mpi_check_status(MPI_Allreduce(MPI_IN_PLACE, &exchangeAll, 1, MPI_INT, MPI_BOR, comm_.get()));
    mpi_check_status(
        MPI_Allreduce(MPI_IN_PLACE, &executionUnitAll, 1, MPI_INT, MPI_BOR, comm_.get()));

    if (exchangeAll != exchangeType || executionUnitAll != executionUnit) {
      errorDetected = 1;
    }

    // check if any rank has detected an error
    mpi_check_status(MPI_Allreduce(MPI_IN_PLACE, &errorDetected, 1, MPI_INT, MPI_SUM, comm_.get()));
    if (errorDetected) {
      throw MPIParameterMismatchError();
    }
  }

  // set number of threads to default omp value if not valid
  if (numThreads < 1) {
    numThreads = omp_get_max_threads();
    numThreads_ = omp_get_max_threads();
  }

  // set default exchange type
  if (exchangeType == SpfftExchangeType::SPFFT_EXCH_DEFAULT) {
    exchangeType = SpfftExchangeType::SPFFT_EXCH_COMPACT_BUFFERED;
    exchangeType_ = SpfftExchangeType::SPFFT_EXCH_COMPACT_BUFFERED;
  }

  // mark as local if comm size is 1
  if (comm_.size() == 1) isLocal_ = true;


  int requiredSize = 0;
  switch (exchangeType) {
    case SpfftExchangeType::SPFFT_EXCH_BUFFERED: {
      decltype(maxNumLocalXYPlanes_) globalMaxNumXYPlanes = 0;
      decltype(maxNumLocalZSticks_) globalMaxNumZSticks = 0;
      MPI_Allreduce(&maxNumLocalXYPlanes_, &globalMaxNumXYPlanes, 1,
                    MPIMatchElementaryType<decltype(maxNumLocalXYPlanes_)>::get(), MPI_MAX, comm);
      MPI_Allreduce(&maxNumLocalZSticks_, &globalMaxNumZSticks, 1,
                    MPIMatchElementaryType<decltype(maxNumLocalZSticks_)>::get(), MPI_MAX, comm);
      requiredSize =
          std::max({globalMaxNumXYPlanes * globalMaxNumZSticks * static_cast<int>(comm_.size() + 1),
                    maxDimX_ * maxDimY_ * maxNumLocalXYPlanes_, maxDimZ_ * maxNumLocalZSticks_});
    } break;
    default: {
      // AUTO or COMPACT_BUFFERED or UNBUFFERED
      requiredSize =
          std::max(maxDimX_ * maxDimY_ * maxNumLocalXYPlanes_, maxDimZ_ * maxNumLocalZSticks_);

    } break;
  }

  // Host
  arrayHost1_ = HostArray<ComplexType>(static_cast<SizeType>(requiredSize));
  arrayHost2_ = HostArray<ComplexType>(static_cast<SizeType>(requiredSize));

  // GPU
  if (executionUnit & SpfftProcessingUnitType::SPFFT_PU_GPU) {
#if (defined(SPFFT_CUDA) || defined(SPFFT_ROCM))
    // store device id
    gpu::check_status(gpu::get_device(&deviceId_));

    arrayHost1_.pin_memory();
    arrayHost2_.pin_memory();
    arrayGPU1_ = GPUArray<typename gpu::fft::ComplexType<ValueType>::type>(
        static_cast<SizeType>(requiredSize));
    arrayGPU2_ = GPUArray<typename gpu::fft::ComplexType<ValueType>::type>(
        static_cast<SizeType>(requiredSize));

    // each transform will resize the work buffer as needed
    fftWorkBuffer_.reset(new GPUArray<char>());
#else
    throw GPUSupportError();
#endif
  }
}
#endif

template <typename T>
GridInternal<T>::GridInternal(const GridInternal<T>& grid)
    : isLocal_(grid.isLocal_),
      executionUnit_(grid.executionUnit_),
      deviceId_(grid.deviceId_),
      numThreads_(grid.numThreads_),
      maxDimX_(grid.maxDimX_),
      maxDimY_(grid.maxDimY_),
      maxDimZ_(grid.maxDimZ_),
      maxNumLocalZSticks_(grid.maxNumLocalZSticks_),
      maxNumLocalXYPlanes_(grid.maxNumLocalXYPlanes_),
      arrayHost1_(grid.arrayHost1_.size()),
      arrayHost2_(grid.arrayHost2_.size()) {
#ifdef SPFFT_MPI
  if (!grid.isLocal_) comm_ = MPICommunicatorHandle(grid.comm_.get());
  exchangeType_ = grid.exchangeType_;
#endif
#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
  if (grid.executionUnit_ & SPFFT_PU_GPU) {
    GPUDeviceGuard(grid.device_id());

    if (grid.arrayGPU1_.size() > 0)
      arrayGPU1_ =
          GPUArray<typename gpu::fft::ComplexType<ValueType>::type>(grid.arrayGPU1_.size());
    if (grid.arrayGPU2_.size() > 0)
      arrayGPU2_ =
          GPUArray<typename gpu::fft::ComplexType<ValueType>::type>(grid.arrayGPU2_.size());
    if (grid.fftWorkBuffer_) fftWorkBuffer_.reset(new GPUArray<char>(grid.fftWorkBuffer_->size()));
  }
#endif
}

// instatiate templates for float and double
template class GridInternal<double>;
#ifdef SPFFT_SINGLE_PRECISION
template class GridInternal<float>;
#endif

}  // namespace spfft
