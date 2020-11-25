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
#ifndef SPFFT_EXECUTION_GPU
#define SPFFT_EXECUTION_GPU

#include <complex>
#include <memory>
#include "compression/compression_gpu.hpp"
#include "compression/indices.hpp"
#include "fft/transform_interface.hpp"
#include "gpu_util/gpu_event_handle.hpp"
#include "gpu_util/gpu_fft_api.hpp"
#include "gpu_util/gpu_stream_handle.hpp"
#include "memory/gpu_array.hpp"
#include "memory/gpu_array_view.hpp"
#include "memory/host_array.hpp"
#include "parameters/parameters.hpp"
#include "spfft/config.h"
#include "spfft/types.h"
#include "symmetry/symmetry.hpp"
#include "transpose/transpose.hpp"
#include "util/common_types.hpp"

#ifdef SPFFT_MPI
#include "mpi_util/mpi_communicator_handle.hpp"
#endif

namespace spfft {

// Controls the execution of the 3D FFT from a compressed format in frequency space and slices in
// space domain. Memory is NOT owned by this class and must remain valid during the lifetime.
template <typename T>
class ExecutionGPU {
public:
  // Initialize a local execution on GPU
  ExecutionGPU(const int numThreads, std::shared_ptr<Parameters> param,
               HostArray<std::complex<T>>& array1, HostArray<std::complex<T>>& array2,
               GPUArray<typename gpu::fft::ComplexType<T>::type>& gpuArray1,
               GPUArray<typename gpu::fft::ComplexType<T>::type>& gpuArray2,
               const std::shared_ptr<GPUArray<char>>& fftWorkBuffer);

#ifdef SPFFT_MPI
  // Initialize a distributed execution on GPU
  ExecutionGPU(MPICommunicatorHandle comm, const SpfftExchangeType exchangeType,
               const int numThreads, std::shared_ptr<Parameters> param,
               HostArray<std::complex<T>>& array1, HostArray<std::complex<T>>& array2,
               GPUArray<typename gpu::fft::ComplexType<T>::type>& gpuArray1,
               GPUArray<typename gpu::fft::ComplexType<T>::type>& gpuArray2,
               const std::shared_ptr<GPUArray<char>>& fftWorkBuffer);
#endif

  // transform forward from a given memory location (Host or GPU).
  // The output is located on the GPU.
  auto forward_z(T* output, const SpfftScalingType scalingType) -> void;
  auto forward_exchange(const bool nonBlockingExchange) -> void;
  auto forward_xy(const T* input) -> void;

  // transform backward into a given memory location (Host or GPU).
  // The input is taken from the GPU.
  auto backward_z(const T* input) -> void;
  auto backward_exchange(const bool nonBlockingExchange) -> void;
  auto backward_xy(T* output) -> void;

  auto synchronize() -> void;

  // The space domain data on Host
  auto space_domain_data_host() -> HostArrayView3D<T>;

  // The space domain data on GPU
  auto space_domain_data_gpu() -> GPUArrayView3D<T>;

private:
  GPUStreamHandle stream_;
  GPUEventHandle event_;
  int numThreads_;
  T scalingFactor_;
  std::unique_ptr<TransformGPU> transformZ_;
  std::unique_ptr<Transpose> transpose_;
  std::unique_ptr<TransformGPU> transformXY_;

  std::unique_ptr<Symmetry> zStickSymmetry_;
  std::unique_ptr<Symmetry> planeSymmetry_;

  std::unique_ptr<CompressionGPU> compression_;

  HostArrayView3D<T> spaceDomainDataExternalHost_;
  GPUArrayView3D<T> spaceDomainDataExternalGPU_;

  GPUArrayView2D<typename gpu::fft::ComplexType<T>::type> freqDomainDataGPU_;
  GPUArrayView1D<T> freqDomainCompressedDataGPU_;
  GPUArrayView3D<typename gpu::fft::ComplexType<T>::type> freqDomainXYGPU_;
};
}  // namespace spfft
#endif
