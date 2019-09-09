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
#if defined(SPFFT_MPI) && (defined(SPFFT_CUDA) || defined(SPFFT_ROCM))
#include <algorithm>
#include <cassert>
#include <complex>
#include <cstring>
#include <utility>
#include <vector>
#include "memory/array_view_utility.hpp"
#include "memory/host_array_view.hpp"
#include "parameters/parameters.hpp"
#include "spfft/exceptions.hpp"
#include "transpose.hpp"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"
#include "util/type_check.hpp"

#include "gpu_util/gpu_fft_api.hpp"
#include "gpu_util/gpu_transfer.hpp"
#include "mpi_util/mpi_check_status.hpp"
#include "mpi_util/mpi_communicator_handle.hpp"
#include "mpi_util/mpi_datatype_handle.hpp"
#include "mpi_util/mpi_match_elementary_type.hpp"
#include "transpose/gpu_kernels/compact_buffered_kernels.hpp"
#include "transpose/transpose_mpi_compact_buffered_gpu.hpp"

namespace spfft {
template <typename T, typename U>
TransposeMPICompactBufferedGPU<T, U>::TransposeMPICompactBufferedGPU(
    const std::shared_ptr<Parameters>& param, MPICommunicatorHandle comm,
    HostArrayView1D<ComplexType> spaceDomainBufferHost,
    GPUArrayView3D<ComplexGPUType> spaceDomainDataGPU,
    GPUArrayView1D<ComplexGPUType> spaceDomainBufferGPU, GPUStreamHandle spaceDomainStream,
    HostArrayView1D<ComplexType> freqDomainBufferHost,
    GPUArrayView2D<ComplexGPUType> freqDomainDataGPU,
    GPUArrayView1D<ComplexGPUType> freqDomainBufferGPU, GPUStreamHandle freqDomainStream)
    : param_(param),
      comm_(std::move(comm)),
      spaceDomainBufferHost_(create_new_type_1d_view<ComplexExchangeType>(
          spaceDomainBufferHost,
          param_->num_xy_planes(comm_.rank()) * param_->total_num_z_sticks())),
      freqDomainBufferHost_(create_new_type_1d_view<ComplexExchangeType>(
          freqDomainBufferHost,
          param_->total_num_xy_planes() * param_->num_z_sticks(comm_.rank()))),
      spaceDomainDataGPU_(spaceDomainDataGPU),
      freqDomainDataGPU_(freqDomainDataGPU),
      spaceDomainBufferGPU_(create_new_type_1d_view<ComplexExchangeGPUType>(
          spaceDomainBufferGPU,
          param_->total_num_z_sticks() * param_->num_xy_planes(comm_.rank()))),
      freqDomainBufferGPU_(create_new_type_1d_view<ComplexExchangeGPUType>(
          freqDomainBufferGPU, param_->num_z_sticks(comm_.rank()) * param_->total_num_xy_planes())),
      spaceDomainStream_(std::move(spaceDomainStream)),
      freqDomainStream_(std::move(freqDomainStream)) {
  assert(param_->dim_y() == spaceDomainDataGPU.dim_mid());
  assert(param_->dim_x_freq() == spaceDomainDataGPU.dim_inner());
  assert(param_->num_xy_planes(comm_.rank()) == spaceDomainDataGPU.dim_outer());
  assert(param_->dim_z() == freqDomainDataGPU.dim_inner());
  assert(param_->num_z_sticks(comm_.rank()) == freqDomainDataGPU.dim_outer());

  assert(spaceDomainBufferGPU.size() >=
         param_->total_num_z_sticks() * param_->num_xy_planes(comm_.rank()));
  assert(spaceDomainBufferHost.size() >=
         param_->total_num_z_sticks() * param_->num_xy_planes(comm_.rank()));
  assert(freqDomainBufferGPU.size() >=
         param_->total_num_xy_planes() * param_->num_z_sticks(comm_.rank()));
  assert(freqDomainBufferHost.size() >=
         param_->total_num_xy_planes() * param_->num_z_sticks(comm_.rank()));

  // assert(disjoint(spaceDomainDataGPU, freqDomainDataGPU));
  assert(disjoint(spaceDomainDataGPU, spaceDomainBufferGPU));
  assert(disjoint(freqDomainDataGPU, freqDomainBufferGPU));
  assert(disjoint(spaceDomainBufferHost, freqDomainBufferHost));
#ifdef SPFFT_GPU_DIRECT
  assert(disjoint(spaceDomainBufferGPU, freqDomainBufferGPU));
#endif

  // create underlying type
  mpiTypeHandle_ = MPIDatatypeHandle::create_contiguous(2, MPIMatchElementaryType<U>::get());

  // prepare mpi parameters
  spaceDomainCount_.resize(comm_.size());
  freqDomainCount_.resize(comm_.size());
  const SizeType numLocalZSticks = param_->num_z_sticks(comm_.rank());
  const SizeType numLocalXYPlanes = param_->num_xy_planes(comm_.rank());
  for (SizeType r = 0; r < (SizeType)comm_.size(); ++r) {
    freqDomainCount_[r] = numLocalZSticks * param_->num_xy_planes(r);
    spaceDomainCount_[r] = param_->num_z_sticks(r) * numLocalXYPlanes;
  }

  spaceDomainDispls_.resize(comm_.size());
  freqDomainDispls_.resize(comm_.size());
  int currentFreqDomainDispls = 0;
  int currentSpaceDomainDispls = 0;
  for (SizeType r = 0; r < (SizeType)comm_.size(); ++r) {
    assert(currentSpaceDomainDispls + spaceDomainCount_[r] <=
           static_cast<int>(spaceDomainBufferHost.size()));
    assert(currentFreqDomainDispls + freqDomainCount_[r] <=
           static_cast<int>(freqDomainBufferHost.size()));
    spaceDomainDispls_[r] = currentSpaceDomainDispls;
    freqDomainDispls_[r] = currentFreqDomainDispls;
    currentSpaceDomainDispls += spaceDomainCount_[r];
    currentFreqDomainDispls += freqDomainCount_[r];
  }

  // copy relevant parameters to gpu
  std::vector<int> numZSticksHost(comm_.size());
  std::vector<int> numXYPlanesHost(comm_.size());
  std::vector<int> xyPlaneOffsetsHost(comm_.size());
  std::vector<int> indicesHost(comm_.size() * param_->max_num_z_sticks());
  for (SizeType r = 0; r < comm_.size(); ++r) {
    numZSticksHost[r] = static_cast<int>(param_->num_z_sticks(r));
    numXYPlanesHost[r] = static_cast<int>(param_->num_xy_planes(r));
    xyPlaneOffsetsHost[r] = static_cast<int>(param_->xy_plane_offset(r));
    const auto zStickXYIndices = param_->z_stick_xy_indices(r);
    for (SizeType i = 0; i < zStickXYIndices.size(); ++i) {
      // transpose stick index
      const int xyIndex = zStickXYIndices(i);
      const int x = xyIndex / param_->dim_y();
      const int y = xyIndex - x * param_->dim_y();
      indicesHost[r * param_->max_num_z_sticks() + i] = y * param_->dim_x_freq() + x;
    }
  }
  numZSticksGPU_ = GPUArray<int>(numZSticksHost.size());
  numXYPlanesGPU_ = GPUArray<int>(numXYPlanesHost.size());
  xyPlaneOffsetsGPU_ = GPUArray<int>(xyPlaneOffsetsHost.size());
  indicesGPU_ = GPUArray<int>(indicesHost.size());

  copy_to_gpu(numZSticksHost, numZSticksGPU_);
  copy_to_gpu(numXYPlanesHost, numXYPlanesGPU_);
  copy_to_gpu(xyPlaneOffsetsHost, xyPlaneOffsetsGPU_);
  copy_to_gpu(indicesHost, indicesGPU_);
}

template <typename T, typename U>
auto TransposeMPICompactBufferedGPU<T, U>::pack_backward() -> void {
  if (freqDomainDataGPU_.size() > 0 && freqDomainBufferGPU_.size() > 0) {
    compact_buffered_pack_backward(freqDomainStream_.get(), param_->max_num_xy_planes(),
                                   create_1d_view(numXYPlanesGPU_, 0, numXYPlanesGPU_.size()),
                                   create_1d_view(xyPlaneOffsetsGPU_, 0, xyPlaneOffsetsGPU_.size()),
                                   freqDomainDataGPU_, freqDomainBufferGPU_);
#ifndef SPFFT_GPU_DIRECT
    copy_from_gpu_async(freqDomainStream_, freqDomainBufferGPU_, freqDomainBufferHost_);
#endif
  }
}

template <typename T, typename U>
auto TransposeMPICompactBufferedGPU<T, U>::unpack_backward() -> void {
  if (spaceDomainDataGPU_.size() > 0) {
    gpu::check_status(gpu::memset_async(
        static_cast<void*>(spaceDomainDataGPU_.data()), 0,
        spaceDomainDataGPU_.size() * sizeof(typename decltype(spaceDomainDataGPU_)::ValueType),
        spaceDomainStream_.get()));
    if (spaceDomainBufferGPU_.size() > 0) {
#ifndef SPFFT_GPU_DIRECT
      copy_to_gpu_async(spaceDomainStream_, spaceDomainBufferHost_, spaceDomainBufferGPU_);
#endif
      compact_buffered_unpack_backward(spaceDomainStream_.get(), param_->max_num_z_sticks(),
                                       create_1d_view(numZSticksGPU_, 0, numZSticksGPU_.size()),
                                       create_1d_view(indicesGPU_, 0, indicesGPU_.size()),
                                       spaceDomainBufferGPU_, spaceDomainDataGPU_);
    }
  }
}

template <typename T, typename U>
auto TransposeMPICompactBufferedGPU<T, U>::exchange_backward_start(const bool nonBlockingExchange)
    -> void {
  assert(omp_get_thread_num() == 0);  // only must thread must be allowed to enter

  gpu::check_status(gpu::stream_synchronize(freqDomainStream_.get()));

  // exchange data
  if (nonBlockingExchange) {
#ifdef SPFFT_GPU_DIRECT
    mpi_check_status(MPI_Ialltoallv(freqDomainBufferGPU_.data(), freqDomainCount_.data(),
                                    freqDomainDispls_.data(), mpiTypeHandle_.get(),
                                    spaceDomainBufferGPU_.data(), spaceDomainCount_.data(),
                                    spaceDomainDispls_.data(), mpiTypeHandle_.get(), comm_.get(),
                                    mpiRequest_.get_and_activate()));
#else
    mpi_check_status(MPI_Ialltoallv(freqDomainBufferHost_.data(), freqDomainCount_.data(),
                                    freqDomainDispls_.data(), mpiTypeHandle_.get(),
                                    spaceDomainBufferHost_.data(), spaceDomainCount_.data(),
                                    spaceDomainDispls_.data(), mpiTypeHandle_.get(), comm_.get(),
                                    mpiRequest_.get_and_activate()));
#endif
  } else {
#ifdef SPFFT_GPU_DIRECT
    mpi_check_status(MPI_Alltoallv(freqDomainBufferGPU_.data(), freqDomainCount_.data(),
                                   freqDomainDispls_.data(), mpiTypeHandle_.get(),
                                   spaceDomainBufferGPU_.data(), spaceDomainCount_.data(),
                                   spaceDomainDispls_.data(), mpiTypeHandle_.get(), comm_.get()));
#else
    mpi_check_status(MPI_Alltoallv(freqDomainBufferHost_.data(), freqDomainCount_.data(),
                                   freqDomainDispls_.data(), mpiTypeHandle_.get(),
                                   spaceDomainBufferHost_.data(), spaceDomainCount_.data(),
                                   spaceDomainDispls_.data(), mpiTypeHandle_.get(), comm_.get()));
#endif
  }
}

template <typename T, typename U>
auto TransposeMPICompactBufferedGPU<T, U>::exchange_backward_finalize() -> void {
  mpiRequest_.wait_if_active();
}

template <typename T, typename U>
auto TransposeMPICompactBufferedGPU<T, U>::pack_forward() -> void {
  if (spaceDomainDataGPU_.size() > 0 && spaceDomainBufferGPU_.size() > 0) {
    compact_buffered_pack_forward(spaceDomainStream_.get(), param_->max_num_z_sticks(),
                                  create_1d_view(numZSticksGPU_, 0, numZSticksGPU_.size()),
                                  create_1d_view(indicesGPU_, 0, indicesGPU_.size()),
                                  spaceDomainDataGPU_, spaceDomainBufferGPU_);

#ifndef SPFFT_GPU_DIRECT
    copy_from_gpu_async(spaceDomainStream_, spaceDomainBufferGPU_, spaceDomainBufferHost_);
#endif
  }
}

template <typename T, typename U>
auto TransposeMPICompactBufferedGPU<T, U>::unpack_forward() -> void {
  if (freqDomainDataGPU_.size() > 0 && freqDomainBufferGPU_.size() > 0) {
#ifndef SPFFT_GPU_DIRECT
    copy_to_gpu_async(freqDomainStream_, freqDomainBufferHost_, freqDomainBufferGPU_);
#endif
    compact_buffered_unpack_forward(
        freqDomainStream_.get(), param_->max_num_xy_planes(),
        create_1d_view(numXYPlanesGPU_, 0, numXYPlanesGPU_.size()),
        create_1d_view(xyPlaneOffsetsGPU_, 0, xyPlaneOffsetsGPU_.size()), freqDomainBufferGPU_,
        freqDomainDataGPU_);
  }
}

template <typename T, typename U>
auto TransposeMPICompactBufferedGPU<T, U>::exchange_forward_start(const bool nonBlockingExchange)
    -> void {
  assert(omp_get_thread_num() == 0);  // only must thread must be allowed to enter

  gpu::check_status(gpu::stream_synchronize(spaceDomainStream_.get()));

  // exchange data
  if (nonBlockingExchange) {
#ifdef SPFFT_GPU_DIRECT
    mpi_check_status(MPI_Ialltoallv(spaceDomainBufferGPU_.data(), spaceDomainCount_.data(),
                                    spaceDomainDispls_.data(), mpiTypeHandle_.get(),
                                    freqDomainBufferGPU_.data(), freqDomainCount_.data(),
                                    freqDomainDispls_.data(), mpiTypeHandle_.get(), comm_.get(),
                                    mpiRequest_.get_and_activate()));
#else
    mpi_check_status(MPI_Ialltoallv(spaceDomainBufferHost_.data(), spaceDomainCount_.data(),
                                    spaceDomainDispls_.data(), mpiTypeHandle_.get(),
                                    freqDomainBufferHost_.data(), freqDomainCount_.data(),
                                    freqDomainDispls_.data(), mpiTypeHandle_.get(), comm_.get(),
                                    mpiRequest_.get_and_activate()));
#endif
  } else {
#ifdef SPFFT_GPU_DIRECT
    mpi_check_status(MPI_Alltoallv(spaceDomainBufferGPU_.data(), spaceDomainCount_.data(),
                                   spaceDomainDispls_.data(), mpiTypeHandle_.get(),
                                   freqDomainBufferGPU_.data(), freqDomainCount_.data(),
                                   freqDomainDispls_.data(), mpiTypeHandle_.get(), comm_.get()));
#else
    mpi_check_status(MPI_Alltoallv(spaceDomainBufferHost_.data(), spaceDomainCount_.data(),
                                   spaceDomainDispls_.data(), mpiTypeHandle_.get(),
                                   freqDomainBufferHost_.data(), freqDomainCount_.data(),
                                   freqDomainDispls_.data(), mpiTypeHandle_.get(), comm_.get()));
#endif
  }
}

template <typename T, typename U>
auto TransposeMPICompactBufferedGPU<T, U>::exchange_forward_finalize() -> void {
  mpiRequest_.wait_if_active();
}

// Instantiate class for float and double
#ifdef SPFFT_SINGLE_PRECISION
template class TransposeMPICompactBufferedGPU<float, float>;
#endif
template class TransposeMPICompactBufferedGPU<double, double>;
template class TransposeMPICompactBufferedGPU<double, float>;
}  // namespace spfft
#endif  // SPFFT_MPI
