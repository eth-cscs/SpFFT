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
#include <algorithm>
#include <cassert>
#include <complex>
#include <cstring>
#include <utility>
#include <vector>
#include "gpu_util/gpu_transfer.hpp"
#include "memory/array_view_utility.hpp"
#include "memory/host_array_view.hpp"
#include "parameters/parameters.hpp"
#include "spfft/exceptions.hpp"
#include "transpose.hpp"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"
#include "util/type_check.hpp"

#ifdef SPFFT_MPI
#include "mpi_util/mpi_check_status.hpp"
#include "mpi_util/mpi_communicator_handle.hpp"
#include "mpi_util/mpi_datatype_handle.hpp"
#include "mpi_util/mpi_match_elementary_type.hpp"
#include "transpose/transpose_mpi_unbuffered_gpu.hpp"

namespace spfft {
template <typename T>
TransposeMPIUnbufferedGPU<T>::TransposeMPIUnbufferedGPU(
    const std::shared_ptr<Parameters>& param, MPICommunicatorHandle comm,
    HostArrayView3D<ComplexType> spaceDomainData,
    GPUArrayView3D<typename gpu::fft::ComplexType<ValueType>::type> spaceDomainDataGPU,
    GPUStreamHandle spaceDomainStream, HostArrayView2D<ComplexType> freqDomainData,
    GPUArrayView2D<typename gpu::fft::ComplexType<ValueType>::type> freqDomainDataGPU,
    GPUStreamHandle freqDomainStream)
    : comm_(std::move(comm)),
      spaceDomainData_(spaceDomainData),
      freqDomainData_(freqDomainData),
      spaceDomainDataGPU_(spaceDomainDataGPU),
      freqDomainDataGPU_(freqDomainDataGPU),
      numLocalXYPlanes_(spaceDomainData.dim_outer()),
      spaceDomainStream_(std::move(spaceDomainStream)),
      freqDomainStream_(std::move(freqDomainStream)) {
  assert(disjoint(spaceDomainData, freqDomainData));
  assert(param->dim_x_freq() == spaceDomainData.dim_inner());
  assert(param->dim_y() == spaceDomainData.dim_mid());
  assert(param->num_xy_planes(comm_.rank()) == spaceDomainData.dim_outer());
  assert(param->dim_z() == freqDomainData.dim_inner());
  assert(param->num_z_sticks(comm_.rank()) == freqDomainData.dim_outer());

  // create underlying type
  MPIDatatypeHandle complexType =
      MPIDatatypeHandle::create_contiguous(2, MPIMatchElementaryType<T>::get());

  // create types in frequency space for each rank:
  // each type represents a fixed length part of every z stick the rank holds
  freqDomainTypeHandles_.reserve(comm_.size());
  freqDomainCount_.reserve(comm_.size());
  freqDomainTypes_.reserve(comm_.size());
  freqDomainDispls_.assign(comm_.size(), 0);

  const SizeType numLocalZSticks = param->num_z_sticks(comm_.rank());
  const SizeType numLocalXYPlanes = param->num_xy_planes(comm_.rank());
  for (SizeType r = 0; r < comm_.size(); ++r) {
    if (param->num_xy_planes(r) > 0 && numLocalZSticks > 0) {
      const int ndims = 2;
      const int arrayOfSizes[] = {(int)numLocalZSticks, (int)freqDomainData_.dim_inner()};
      const int arrayOfSubsizes[] = {(int)numLocalZSticks, (int)param->num_xy_planes(r)};
      const int arrayOfStarts[] = {(int)0, (int)param->xy_plane_offset(r)};
      const int order = MPI_ORDER_C;

      freqDomainCount_.emplace_back(1);
      freqDomainTypeHandles_.emplace_back(MPIDatatypeHandle::create_subarray(
          ndims, arrayOfSizes, arrayOfSubsizes, arrayOfStarts, order, complexType.get()));
      freqDomainTypes_.emplace_back(freqDomainTypeHandles_.back().get());
    } else {
      freqDomainCount_.emplace_back(0);
      freqDomainTypeHandles_.emplace_back(complexType);
      freqDomainTypes_.emplace_back(freqDomainTypeHandles_.back().get());
    }
  }

  // create types in space domain for each rank:
  // each type represents a batch of partial z sticks with inner stride dimX*dimY and placed
  // according to the assosiated x/y indices
  std::vector<int> indexedBlocklengths;
  std::vector<MPI_Aint> indexedDispls;

  spaceDomainTypes_.reserve(comm_.size());
  spaceDomainCount_.reserve(comm_.size());
  spaceDomainDispls_.assign(comm_.size(), 0);
  for (SizeType r = 0; r < comm_.size(); ++r) {
    if (param->num_z_sticks(r) > 0 && numLocalXYPlanes > 0) {
      // data type for single z stick part
      MPIDatatypeHandle stridedZStickType = MPIDatatypeHandle::create_vector(
          numLocalXYPlanes, 1, spaceDomainData_.dim_inner() * spaceDomainData_.dim_mid(),
          complexType.get());

      const auto zStickXYIndices = param->z_stick_xy_indices(r);

      indexedBlocklengths.resize(zStickXYIndices.size(), 1);
      indexedDispls.resize(zStickXYIndices.size());
      // displacements of all z stick parts to be send to current rank
      for (SizeType idxZStick = 0; idxZStick < zStickXYIndices.size(); ++idxZStick) {
        // transpose stick index
        const int xyIndex = zStickXYIndices(idxZStick);
        const int x = xyIndex / param->dim_y();
        const int y = xyIndex - x * param->dim_y();

        indexedDispls[idxZStick] = 2 * sizeof(T) * (y * param->dim_x_freq() + x);
      }

      spaceDomainCount_.emplace_back(1);
      spaceDomainTypeHandles_.emplace_back(
          MPIDatatypeHandle::create_hindexed(zStickXYIndices.size(), indexedBlocklengths.data(),
                                             indexedDispls.data(), stridedZStickType.get()));
      spaceDomainTypes_.emplace_back(spaceDomainTypeHandles_.back().get());
    } else {
      spaceDomainCount_.emplace_back(0);
      spaceDomainTypeHandles_.emplace_back(complexType);
      spaceDomainTypes_.emplace_back(complexType.get());
    }
  }
}

template <typename T>
auto TransposeMPIUnbufferedGPU<T>::pack_backward() -> void {
#ifdef SPFFT_GPU_DIRECT
  gpu::check_status(gpu::memset_async(
      static_cast<void*>(spaceDomainDataGPU_.data()), 0,
      spaceDomainDataGPU_.size() * sizeof(typename decltype(spaceDomainDataGPU_)::ValueType),
      spaceDomainStream_.get()));
#else
  copy_from_gpu_async(freqDomainStream_, freqDomainDataGPU_, freqDomainData_);
  // zero target data location (not all values are overwritten upon unpacking)
  std::memset(static_cast<void*>(spaceDomainData_.data()), 0,
              sizeof(typename decltype(spaceDomainData_)::ValueType) * spaceDomainData_.size());
#endif
}

template <typename T>
auto TransposeMPIUnbufferedGPU<T>::unpack_backward() -> void {
#ifndef SPFFT_GPU_DIRECT
  copy_to_gpu_async(spaceDomainStream_, spaceDomainData_, spaceDomainDataGPU_);
#endif
}

template <typename T>
auto TransposeMPIUnbufferedGPU<T>::exchange_backward_start(const bool nonBlockingExchange) -> void {
  assert(omp_get_thread_num() == 0);  // only must thread must be allowed to enter

  gpu::check_status(gpu::stream_synchronize(freqDomainStream_.get()));

  if (nonBlockingExchange) {
#ifdef SPFFT_GPU_DIRECT
    mpi_check_status(MPI_Ialltoallw(freqDomainDataGPU_.data(), freqDomainCount_.data(),
                                    freqDomainDispls_.data(), freqDomainTypes_.data(),
                                    spaceDomainDataGPU_.data(), spaceDomainCount_.data(),
                                    spaceDomainDispls_.data(), spaceDomainTypes_.data(),
                                    comm_.get(), mpiRequest_.get_and_activate()));
#else
    mpi_check_status(MPI_Ialltoallw(freqDomainData_.data(), freqDomainCount_.data(),
                                    freqDomainDispls_.data(), freqDomainTypes_.data(),
                                    spaceDomainData_.data(), spaceDomainCount_.data(),
                                    spaceDomainDispls_.data(), spaceDomainTypes_.data(),
                                    comm_.get(), mpiRequest_.get_and_activate()));
#endif
  } else {
#ifdef SPFFT_GPU_DIRECT
    mpi_check_status(
        MPI_Alltoallw(freqDomainDataGPU_.data(), freqDomainCount_.data(), freqDomainDispls_.data(),
                      freqDomainTypes_.data(), spaceDomainDataGPU_.data(), spaceDomainCount_.data(),
                      spaceDomainDispls_.data(), spaceDomainTypes_.data(), comm_.get()));
#else
    mpi_check_status(
        MPI_Alltoallw(freqDomainData_.data(), freqDomainCount_.data(), freqDomainDispls_.data(),
                      freqDomainTypes_.data(), spaceDomainData_.data(), spaceDomainCount_.data(),
                      spaceDomainDispls_.data(), spaceDomainTypes_.data(), comm_.get()));
#endif
  }
}

template <typename T>
auto TransposeMPIUnbufferedGPU<T>::exchange_backward_finalize() -> void {
  mpiRequest_.wait_if_active();
}

template <typename T>
auto TransposeMPIUnbufferedGPU<T>::exchange_forward_start(const bool nonBlockingExchange) -> void {
  assert(omp_get_thread_num() == 0);  // only must thread must be allowed to enter

  gpu::check_status(gpu::stream_synchronize(spaceDomainStream_.get()));

  if (nonBlockingExchange) {
#ifdef SPFFT_GPU_DIRECT
    mpi_check_status(MPI_Ialltoallw(spaceDomainDataGPU_.data(), spaceDomainCount_.data(),
                                    spaceDomainDispls_.data(), spaceDomainTypes_.data(),
                                    freqDomainDataGPU_.data(), freqDomainCount_.data(),
                                    freqDomainDispls_.data(), freqDomainTypes_.data(), comm_.get(),
                                    mpiRequest_.get_and_activate()));
#else
    mpi_check_status(MPI_Ialltoallw(spaceDomainData_.data(), spaceDomainCount_.data(),
                                    spaceDomainDispls_.data(), spaceDomainTypes_.data(),
                                    freqDomainData_.data(), freqDomainCount_.data(),
                                    freqDomainDispls_.data(), freqDomainTypes_.data(), comm_.get(),
                                    mpiRequest_.get_and_activate()));
#endif
  } else {
#ifdef SPFFT_GPU_DIRECT
    mpi_check_status(MPI_Alltoallw(spaceDomainDataGPU_.data(), spaceDomainCount_.data(),
                                   spaceDomainDispls_.data(), spaceDomainTypes_.data(),
                                   freqDomainDataGPU_.data(), freqDomainCount_.data(),
                                   freqDomainDispls_.data(), freqDomainTypes_.data(), comm_.get()));
#else
    mpi_check_status(MPI_Alltoallw(spaceDomainData_.data(), spaceDomainCount_.data(),
                                   spaceDomainDispls_.data(), spaceDomainTypes_.data(),
                                   freqDomainData_.data(), freqDomainCount_.data(),
                                   freqDomainDispls_.data(), freqDomainTypes_.data(), comm_.get()));
#endif
  }
}

template <typename T>
auto TransposeMPIUnbufferedGPU<T>::exchange_forward_finalize() -> void {
  mpiRequest_.wait_if_active();
}

template <typename T>
auto TransposeMPIUnbufferedGPU<T>::pack_forward() -> void {
#ifndef SPFFT_GPU_DIRECT
  copy_from_gpu_async(spaceDomainStream_, spaceDomainDataGPU_, spaceDomainData_);
#endif
}

template <typename T>
auto TransposeMPIUnbufferedGPU<T>::unpack_forward() -> void {
#ifndef SPFFT_GPU_DIRECT
  copy_to_gpu_async(freqDomainStream_, freqDomainData_, freqDomainDataGPU_);
#endif
}

// Instantiate class for float and double
#ifdef SPFFT_SINGLE_PRECISION
template class TransposeMPIUnbufferedGPU<float>;
#endif
template class TransposeMPIUnbufferedGPU<double>;
}  // namespace spfft
#endif  // SPFFT_MPI
