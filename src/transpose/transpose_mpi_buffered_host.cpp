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
#include "transpose/transpose_mpi_buffered_host.hpp"

namespace spfft {
template <typename T, typename U>
TransposeMPIBufferedHost<T, U>::TransposeMPIBufferedHost(
    const std::shared_ptr<Parameters>& param, MPICommunicatorHandle comm,
    HostArrayView3D<ComplexType> spaceDomainData, HostArrayView2D<ComplexType> freqDomainData,
    HostArrayView1D<ComplexType> spaceDomainBuffer, HostArrayView1D<ComplexType> freqDomainBuffer)
    : param_(param),
      comm_(std::move(comm)),
      spaceDomainData_(spaceDomainData),
      freqDomainData_(freqDomainData),
      spaceDomainBuffer_(create_new_type_1d_view<ComplexExchangeType>(spaceDomainBuffer,
                                                                      spaceDomainBuffer.size())),
      freqDomainBuffer_(
          create_new_type_1d_view<ComplexExchangeType>(freqDomainBuffer, freqDomainBuffer.size())) {
  // assert(param_->dim_x_freq() == spaceDomainData.dim_mid());
  assert(param_->dim_y() == spaceDomainData.dim_inner());
  assert(param_->num_xy_planes(comm_.rank()) == spaceDomainData.dim_outer());
  assert(param_->dim_z() == freqDomainData.dim_inner());
  assert(param_->num_z_sticks(comm_.rank()) == freqDomainData.dim_outer());

  assert(spaceDomainBuffer.size() >=
         param_->max_num_xy_planes() * param_->max_num_z_sticks() * comm_.size());
  assert(freqDomainBuffer.size() >=
         param_->max_num_xy_planes() * param_->max_num_z_sticks() * comm_.size());

  assert(disjoint(spaceDomainData, freqDomainData));
  assert(disjoint(spaceDomainData, spaceDomainBuffer));
  assert(disjoint(freqDomainData, freqDomainBuffer));
  assert(disjoint(spaceDomainBuffer, freqDomainBuffer));

  // create underlying type
  mpiTypeHandle_ = MPIDatatypeHandle::create_contiguous(2, MPIMatchElementaryType<U>::get());
}

template <typename T, typename U>
auto TransposeMPIBufferedHost<T, U>::pack_backward() -> void {
  auto freqDomainBuffer3d = create_3d_view(freqDomainBuffer_, 0, comm_.size(),
                                           param_->max_num_z_sticks(), param_->max_num_xy_planes());
  // transpose locally from (numLocalZSticks, dimZ) to (dimZ, numLocalZSticks) with spacing
  // between ranks
  for (SizeType r = 0; r < static_cast<SizeType>(comm_.size()); ++r) {
    const auto xyPlaneOffset = param_->xy_plane_offset(r);
    SPFFT_OMP_PRAGMA("omp for schedule(static) nowait")
    for (SizeType zStickIndex = 0; zStickIndex < freqDomainData_.dim_outer(); ++zStickIndex) {
      for (SizeType xyPlaneIndex = 0; xyPlaneIndex < param_->num_xy_planes(r); ++xyPlaneIndex) {
        freqDomainBuffer3d(r, zStickIndex, xyPlaneIndex) =
            freqDomainData_(zStickIndex, xyPlaneIndex + xyPlaneOffset);
      }
    }
  }
  SPFFT_OMP_PRAGMA("omp barrier")
}

template <typename T, typename U>
auto TransposeMPIBufferedHost<T, U>::unpack_backward() -> void {
  // zero target data location (not all values are overwritten upon unpacking)
  SPFFT_OMP_PRAGMA("omp for schedule(static)") // implicit barrier
  for (SizeType z = 0; z < spaceDomainData_.dim_outer(); ++z) {
    std::memset(static_cast<void*>(&spaceDomainData_(z, 0, 0)), 0,
                sizeof(typename decltype(spaceDomainData_)::ValueType) *
                    spaceDomainData_.dim_inner() * spaceDomainData_.dim_mid());
  }

  auto spaceDomainDataFlat =
      create_2d_view(spaceDomainData_, 0, spaceDomainData_.dim_outer(),
                     spaceDomainData_.dim_mid() * spaceDomainData_.dim_inner());

  // unpack from (numZSticksTotal, numLocalXYPlanes) to (numLocalXYPlanes, dimX, dimY)
  const auto numLocalXYPlanes = param_->num_xy_planes(comm_.rank());
  for (SizeType r = 0; r < (SizeType)comm_.size(); ++r) {
    const auto zStickXYIndices = param_->z_stick_xy_indices(r);
    // take care with unsigned type
    const SizeType unrolledLoopEnd = zStickXYIndices.size() < 4 ? 0 : zStickXYIndices.size() - 3;

    auto spaceDomainBuffer2d = create_2d_view(
        spaceDomainBuffer_, r * param_->max_num_xy_planes() * param_->max_num_z_sticks(),
        param_->max_num_z_sticks(), param_->max_num_xy_planes());

    SPFFT_OMP_PRAGMA("omp for schedule(static) nowait")
    for (SizeType zStickIndex = 0; zStickIndex < unrolledLoopEnd; zStickIndex += 4) {
      // manual loop unrolling for better performance
      const SizeType xyIndex1 = zStickXYIndices(zStickIndex);
      const SizeType xyIndex2 = zStickXYIndices(zStickIndex + 1);
      const SizeType xyIndex3 = zStickXYIndices(zStickIndex + 2);
      const SizeType xyIndex4 = zStickXYIndices(zStickIndex + 3);
      for (SizeType zIndex = 0; zIndex < numLocalXYPlanes; ++zIndex) {
        spaceDomainDataFlat(zIndex, xyIndex1) = spaceDomainBuffer2d(zStickIndex, zIndex);
        spaceDomainDataFlat(zIndex, xyIndex2) = spaceDomainBuffer2d(zStickIndex + 1, zIndex);
        spaceDomainDataFlat(zIndex, xyIndex3) = spaceDomainBuffer2d(zStickIndex + 2, zIndex);
        spaceDomainDataFlat(zIndex, xyIndex4) = spaceDomainBuffer2d(zStickIndex + 3, zIndex);
      }
    }
    SPFFT_OMP_PRAGMA("omp for schedule(static) nowait")
    for (SizeType zStickIndex = unrolledLoopEnd; zStickIndex < zStickXYIndices.size();
         zStickIndex += 1) {
      const SizeType xyIndex = zStickXYIndices(zStickIndex);
      for (SizeType zIndex = 0; zIndex < numLocalXYPlanes; ++zIndex) {
        spaceDomainDataFlat(zIndex, xyIndex) = spaceDomainBuffer2d(zStickIndex, zIndex);
      }
    }
  }
  SPFFT_OMP_PRAGMA("omp barrier")
}

template <typename T, typename U>
auto TransposeMPIBufferedHost<T, U>::exchange_backward_start(const bool nonBlockingExchange)
    -> void {
  assert(omp_get_thread_num() == 0); // only master thread must be allowed to enter

  // exchange data
  if (nonBlockingExchange) {
    mpi_check_status(MPI_Ialltoall(
        freqDomainBuffer_.data(), param_->max_num_z_sticks() * param_->max_num_xy_planes(),
        mpiTypeHandle_.get(), spaceDomainBuffer_.data(),
        param_->max_num_z_sticks() * param_->max_num_xy_planes(), mpiTypeHandle_.get(), comm_.get(),
        mpiRequest_.get_and_activate()));
  } else {
    mpi_check_status(MPI_Alltoall(freqDomainBuffer_.data(),
                                  param_->max_num_z_sticks() * param_->max_num_xy_planes(),
                                  mpiTypeHandle_.get(), spaceDomainBuffer_.data(),
                                  param_->max_num_z_sticks() * param_->max_num_xy_planes(),
                                  mpiTypeHandle_.get(), comm_.get()));
  }
}

template <typename T, typename U>
auto TransposeMPIBufferedHost<T, U>::exchange_backward_finalize() -> void {
  mpiRequest_.wait_if_active();
}

template <typename T, typename U>
auto TransposeMPIBufferedHost<T, U>::pack_forward() -> void {
  auto spaceDomainDataFlat =
      create_2d_view(spaceDomainData_, 0, spaceDomainData_.dim_outer(),
                     spaceDomainData_.dim_mid() * spaceDomainData_.dim_inner());

  // pack from (numLocalXYPlanes, dimX, dimY) to (numZSticksTotal, numLocalXYPlanes)
  const auto numLocalXYPlanes = param_->num_xy_planes(comm_.rank());
  for (SizeType r = 0; r < (SizeType)comm_.size(); ++r) {
    const auto zStickXYIndices = param_->z_stick_xy_indices(r);
    // take care with unsigned type
    const SizeType unrolledLoopEnd = zStickXYIndices.size() < 4 ? 0 : zStickXYIndices.size() - 3;

    auto spaceDomainBuffer2d = create_2d_view(
        spaceDomainBuffer_, r * param_->max_num_xy_planes() * param_->max_num_z_sticks(),
        param_->max_num_z_sticks(), param_->max_num_xy_planes());

    SPFFT_OMP_PRAGMA("omp for schedule(static) nowait")
    for (SizeType zStickIndex = 0; zStickIndex < unrolledLoopEnd; zStickIndex += 4) {
      // manual loop unrolling for better performance
      const SizeType xyIndex1 = zStickXYIndices(zStickIndex);
      const SizeType xyIndex2 = zStickXYIndices(zStickIndex + 1);
      const SizeType xyIndex3 = zStickXYIndices(zStickIndex + 2);
      const SizeType xyIndex4 = zStickXYIndices(zStickIndex + 3);
      for (SizeType zIndex = 0; zIndex < numLocalXYPlanes; ++zIndex) {
        spaceDomainBuffer2d(zStickIndex, zIndex) = spaceDomainDataFlat(zIndex, xyIndex1);
        spaceDomainBuffer2d(zStickIndex + 1, zIndex) = spaceDomainDataFlat(zIndex, xyIndex2);
        spaceDomainBuffer2d(zStickIndex + 2, zIndex) = spaceDomainDataFlat(zIndex, xyIndex3);
        spaceDomainBuffer2d(zStickIndex + 3, zIndex) = spaceDomainDataFlat(zIndex, xyIndex4);
      }
    }
    SPFFT_OMP_PRAGMA("omp for schedule(static) nowait")
    for (SizeType zStickIndex = unrolledLoopEnd; zStickIndex < zStickXYIndices.size();
         zStickIndex += 1) {
      const SizeType xyIndex = zStickXYIndices(zStickIndex);
      for (SizeType zIndex = 0; zIndex < numLocalXYPlanes; ++zIndex) {
        spaceDomainBuffer2d(zStickIndex, zIndex) = spaceDomainDataFlat(zIndex, xyIndex);
      }
    }
  }
  SPFFT_OMP_PRAGMA("omp barrier")
}

template <typename T, typename U>
auto TransposeMPIBufferedHost<T, U>::unpack_forward() -> void {
  auto freqDomainBuffer3d = create_3d_view(freqDomainBuffer_, 0, comm_.size(),
                                           param_->max_num_z_sticks(), param_->max_num_xy_planes());
  for (SizeType r = 0; r < static_cast<SizeType>(comm_.size()); ++r) {
    const auto xyPlaneOffset = param_->xy_plane_offset(r);
    SPFFT_OMP_PRAGMA("omp for schedule(static) nowait")
    for (SizeType zStickIndex = 0; zStickIndex < freqDomainData_.dim_outer(); ++zStickIndex) {
      for (SizeType xyPlaneIndex = 0; xyPlaneIndex < param_->num_xy_planes(r); ++xyPlaneIndex) {
        freqDomainData_(zStickIndex, xyPlaneIndex + xyPlaneOffset) =
            freqDomainBuffer3d(r, zStickIndex, xyPlaneIndex);
      }
    }
  }
  SPFFT_OMP_PRAGMA("omp barrier")
}

template <typename T, typename U>
auto TransposeMPIBufferedHost<T, U>::exchange_forward_start(const bool nonBlockingExchange)
    -> void {
  assert(omp_get_thread_num() == 0); // only master thread must be allowed to enter

  // exchange data
  if (nonBlockingExchange) {
    mpi_check_status(MPI_Ialltoall(
        spaceDomainBuffer_.data(), param_->max_num_z_sticks() * param_->max_num_xy_planes(),
        mpiTypeHandle_.get(), freqDomainBuffer_.data(),
        param_->max_num_z_sticks() * param_->max_num_xy_planes(), mpiTypeHandle_.get(), comm_.get(),
        mpiRequest_.get_and_activate()));
  } else {
    mpi_check_status(MPI_Alltoall(spaceDomainBuffer_.data(),
                                  param_->max_num_z_sticks() * param_->max_num_xy_planes(),
                                  mpiTypeHandle_.get(), freqDomainBuffer_.data(),
                                  param_->max_num_z_sticks() * param_->max_num_xy_planes(),
                                  mpiTypeHandle_.get(), comm_.get()));
  }
}

template <typename T, typename U>
auto TransposeMPIBufferedHost<T, U>::exchange_forward_finalize() -> void {
  mpiRequest_.wait_if_active();
}

// Instantiate class for float and double
#ifdef SPFFT_SINGLE_PRECISION
template class TransposeMPIBufferedHost<float, float>;
#endif
template class TransposeMPIBufferedHost<double, double>;
template class TransposeMPIBufferedHost<double, float>;
} // namespace spfft
#endif // SPFFT_MPI
