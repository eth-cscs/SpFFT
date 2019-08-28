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
#ifndef SPFFT_TRANSPOSE_MPI_UNBUFFERED_HOST_HPP
#define SPFFT_TRANSPOSE_MPI_UNBUFFERED_HOST_HPP

#include <complex>
#include <memory>
#include "memory/host_array_view.hpp"
#include "parameters/parameters.hpp"
#include "spfft/config.h"
#include "transpose.hpp"
#include "util/common_types.hpp"
#include "util/type_check.hpp"

#ifdef SPFFT_MPI
#include "mpi_util/mpi_communicator_handle.hpp"
#include "mpi_util/mpi_datatype_handle.hpp"
#include "mpi_util/mpi_request_handle.hpp"

namespace spfft {
template <typename T>
class TransposeMPIUnbufferedHost : public Transpose {
  static_assert(IsFloatOrDouble<T>::value, "Type T must be float or double");
  using ValueType = T;
  using ComplexType = std::complex<T>;

public:
  TransposeMPIUnbufferedHost(const std::shared_ptr<Parameters>& param, MPICommunicatorHandle comm,
                             HostArrayView3D<ComplexType> spaceDomainData,
                             HostArrayView2D<ComplexType> freqDomainData);

  auto exchange_backward_start(const bool nonBlockingExchange) -> void override;
  auto exchange_backward_finalize() -> void override;

  auto exchange_forward_start(const bool nonBlockingExchange) -> void override;
  auto exchange_forward_finalize() -> void override;

private:
  MPICommunicatorHandle comm_;
  MPIRequestHandle mpiRequest_;

  HostArrayView3D<ComplexType> spaceDomainData_;
  HostArrayView2D<ComplexType> freqDomainData_;

  SizeType numLocalXYPlanes_;

  std::vector<MPIDatatypeHandle> freqDomainTypeHandles_;
  std::vector<MPI_Datatype> freqDomainTypes_;
  std::vector<int> freqDomainDispls_;
  std::vector<int> freqDomainCount_;
  std::vector<MPIDatatypeHandle> spaceDomainTypeHandles_;
  std::vector<MPI_Datatype> spaceDomainTypes_;
  std::vector<int> spaceDomainDispls_;
  std::vector<int> spaceDomainCount_;
};

} // namespace spfft
#endif // SPFFT_MPI
#endif
