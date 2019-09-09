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
#ifndef SPFFT_EXECUTION_HOST_HPP
#define SPFFT_EXECUTION_HOST_HPP

#include <complex>
#include <memory>
#include <tuple>
#include "compression/compression_host.hpp"
#include "compression/indices.hpp"
#include "fft/transform_interface.hpp"
#include "memory/host_array.hpp"
#include "memory/host_array_view.hpp"
#include "parameters/parameters.hpp"
#include "spfft/config.h"
#include "spfft/types.h"
#include "symmetry/symmetry.hpp"
#include "timing/timing.hpp"
#include "transpose/transpose.hpp"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"

#ifdef SPFFT_MPI
#include "mpi_util/mpi_init_handle.hpp"
#endif

namespace spfft {

// Controls the execution of the 3D FFT from a compressed format in frequency space and slices in
// space domain. Memory is NOT owned by this class and must remain valid during the lifetime.
template <typename T>
class ExecutionHost {
public:
  // Initialize a local execution on Host
  ExecutionHost(const int numThreads, std::shared_ptr<Parameters> param,
                HostArray<std::complex<T>>& array1, HostArray<std::complex<T>>& array2);

#ifdef SPFFT_MPI
  // Initialize a distributed execution on Host
  ExecutionHost(MPICommunicatorHandle comm, const SpfftExchangeType exchangeType,
                const int numThreads, std::shared_ptr<Parameters> param,
                HostArray<std::complex<T>>& array1, HostArray<std::complex<T>>& array2);
#endif

  // Transform forward
  auto forward_z(T* output, const SpfftScalingType scalingType) -> void;
  auto forward_exchange(const bool nonBlockingExchange) -> void;
  auto forward_xy() -> void;

  // Transform backward
  auto backward_z(const T* input) -> void;
  auto backward_exchange(const bool nonBlockingExchange) -> void;
  auto backward_xy() -> void;

  // Access the space domain data
  auto space_domain_data() -> HostArrayView3D<T>;

private:
  int numThreads_;
  T scalingFactor_;
  std::unique_ptr<TransformHost> transformZBackward_;
  std::unique_ptr<TransformHost> transformZForward_;
  std::unique_ptr<TransformHost> transformYBackward_;
  std::unique_ptr<TransformHost> transformYForward_;
  std::unique_ptr<TransformHost> transformXBackward_;
  std::unique_ptr<TransformHost> transformXForward_;

  std::unique_ptr<Transpose> transpose_;

  std::unique_ptr<Symmetry> zStickSymmetry_;
  std::unique_ptr<Symmetry> planeSymmetry_;

  std::unique_ptr<CompressionHost> compression_;

  HostArrayView3D<T> spaceDomainDataExternal_;
  HostArrayView2D<std::complex<T>> freqDomainData_;
};
}  // namespace spfft
#endif
