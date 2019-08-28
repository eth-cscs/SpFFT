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
#ifndef SPFFT_GRID_INTERNAL_HPP
#define SPFFT_GRID_INTERNAL_HPP
#include "spfft/config.h"

#include <algorithm>
#include <complex>
#include <memory>
#include "memory/host_array.hpp"
#include "spfft/types.h"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"
#include "util/type_check.hpp"

#ifdef SPFFT_MPI
#include <mpi.h>
#include "mpi_util/mpi_communicator_handle.hpp"
#endif

#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
#include "gpu_util/gpu_fft_api.hpp"
#include "memory/gpu_array.hpp"
#endif

namespace spfft {

template <typename T>
class GridInternal {
public:
  static_assert(IsFloatOrDouble<T>::value, "Type T must be float or double");
  using ValueType = T;
  using ComplexType = std::complex<T>;

  GridInternal(int maxDimX, int maxDimY, int maxDimZ, int maxNumLocalZSticks,
               SpfftProcessingUnitType executionUnit, int numThreads);

#ifdef SPFFT_MPI
  GridInternal(int maxDimX, int maxDimY, int maxDimZ, int maxNumLocalZSticks,
               int maxNumLocalXYPlanes, SpfftProcessingUnitType executionUnit, int numThreads,
               MPI_Comm comm, SpfftExchangeType exchangeType);
#endif

  GridInternal(const GridInternal<T>& grid);

  GridInternal(GridInternal<T>&&) = default;

  inline GridInternal& operator=(const GridInternal<T>& grid) {
    *this = GridInternal(grid);
    return *this;
  }

  inline GridInternal& operator=(GridInternal<T>&&) = default;

  inline auto max_dim_x() const noexcept -> int { return maxDimX_; }

  inline auto max_dim_y() const noexcept -> int { return maxDimY_; }

  inline auto max_dim_z() const noexcept -> int { return maxDimZ_; }

  inline auto max_num_local_z_columns() const noexcept -> int { return maxNumLocalZSticks_; }

  inline auto max_num_local_xy_planes() const noexcept -> int { return maxNumLocalXYPlanes_; }

  inline auto device_id() const noexcept -> int { return deviceId_; }

  inline auto num_threads() const noexcept -> int { return numThreads_; }

  inline auto array_host_1() -> HostArray<ComplexType>& { return arrayHost1_; }

  inline auto array_host_2() -> HostArray<ComplexType>& { return arrayHost2_; }

  inline auto processing_unit() -> SpfftProcessingUnitType { return executionUnit_; }

  inline auto local() -> bool { return isLocal_; }

#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
  inline auto array_gpu_1() -> GPUArray<typename gpu::fft::ComplexType<ValueType>::type>& {
    return arrayGPU1_;
  }

  inline auto array_gpu_2() -> GPUArray<typename gpu::fft::ComplexType<ValueType>::type>& {
    return arrayGPU2_;
  }

  inline auto fft_work_buffer() -> const std::shared_ptr<GPUArray<char>>& {
    assert(fftWorkBuffer_);
    return fftWorkBuffer_;
  }
#endif

#ifdef SPFFT_MPI
  inline auto communicator() const -> const MPICommunicatorHandle& { return comm_; }

  inline auto exchange_type() const -> SpfftExchangeType { return exchangeType_; }
#endif

private:
  bool isLocal_;
  SpfftProcessingUnitType executionUnit_;
  int deviceId_, numThreads_;
  int maxDimX_, maxDimY_, maxDimZ_, maxNumLocalZSticks_, maxNumLocalXYPlanes_;

  HostArray<ComplexType> arrayHost1_;
  HostArray<ComplexType> arrayHost2_;

#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
  GPUArray<typename gpu::fft::ComplexType<ValueType>::type> arrayGPU1_;
  GPUArray<typename gpu::fft::ComplexType<ValueType>::type> arrayGPU2_;
  std::shared_ptr<GPUArray<char>> fftWorkBuffer_;
#endif

#ifdef SPFFT_MPI
  MPICommunicatorHandle comm_;
  SpfftExchangeType exchangeType_;
#endif
};

} // namespace spfft
#endif
