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

#include "execution/execution_host.hpp"
#include "compression/indices.hpp"
#include "fft/transform_1d_host.hpp"
#include "fft/transform_real_1d_host.hpp"
#include "memory/array_view_utility.hpp"
#include "memory/host_array_view.hpp"
#include "spfft/exceptions.hpp"
#include "symmetry/symmetry_host.hpp"
#include "timing/timing.hpp"
#include "transpose/transpose_host.hpp"
#include "util/common_types.hpp"

#ifdef SPFFT_MPI
#include "transpose/transpose_mpi_buffered_host.hpp"
#include "transpose/transpose_mpi_compact_buffered_host.hpp"
#include "transpose/transpose_mpi_unbuffered_host.hpp"
#endif

namespace spfft {

template <typename T>
ExecutionHost<T>::ExecutionHost(const int numThreads, std::shared_ptr<Parameters> param,
                                HostArray<std::complex<T>>& array1,
                                HostArray<std::complex<T>>& array2)
    : numThreads_(numThreads),
      scalingFactor_(static_cast<T>(
          1.0 / static_cast<double>(param->dim_x() * param->dim_y() * param->dim_z()))) {
  HOST_TIMING_SCOPED("Execution init");
  const SizeType numLocalZSticks = param->num_z_sticks(0);
  const SizeType numLocalXYPlanes = param->num_xy_planes(0);
  std::set<SizeType> uniqueXIndices;
  for (const auto& xyIndex : param->z_stick_xy_indices(0)) {
    uniqueXIndices.emplace(static_cast<SizeType>(xyIndex / param->dim_y()));
  }

  auto freqDomainZ3D = create_3d_view(array1, 0, 1, numLocalZSticks, param->dim_z());
  freqDomainData_ = create_2d_view(freqDomainZ3D, 0, numLocalZSticks, param->dim_z());
  freqDomainXY_ = create_3d_view(array2, 0, param->dim_z(), param->dim_x_freq(), param->dim_y());

  transpose_.reset(new TransposeHost<T>(param, freqDomainXY_, freqDomainData_));

  if (param->local_value_indices().size() > 0) {
    compression_.reset(new CompressionHost(param));
  }

  if (numLocalZSticks > 0) {
    // Z
    transformZBackward_.reset(new Transform1DPlanesHost<T>(freqDomainZ3D, freqDomainZ3D, false,
                                                           false, FFTW_BACKWARD, numThreads));
    transformZForward_.reset(new Transform1DPlanesHost<T>(freqDomainZ3D, freqDomainZ3D, false,
                                                          false, FFTW_FORWARD, numThreads));
  }

  if (numLocalXYPlanes > 0) {
    // Y
    transformYBackward_.reset(new Transform1DVerticalHost<T>(freqDomainXY_, freqDomainXY_, false,
                                                             false, FFTW_BACKWARD, uniqueXIndices));
    transformYForward_.reset(new Transform1DVerticalHost<T>(freqDomainXY_, freqDomainXY_, false,
                                                            false, FFTW_FORWARD, uniqueXIndices));

    // X
    if (param->transform_type() == SPFFT_TRANS_R2C) {
      if (param->zero_zero_stick_index() < param->num_z_sticks(0)) {
        zStickSymmetry_.reset(new StickSymmetryHost<T>(HostArrayView1D<std::complex<T>>(
            &freqDomainData_(param->zero_zero_stick_index(), 0), freqDomainData_.dim_inner(),
            freqDomainData_.pinned())));
      }

      planeSymmetry_.reset(new PlaneSymmetryHost<T>(freqDomainXY_));

      spaceDomainDataExternal_ =
          create_new_type_3d_view<T>(array1, param->dim_z(), param->dim_y(), param->dim_x());
      transformXBackward_.reset(new C2RTransform1DPlanesHost<T>(
          freqDomainXY_, spaceDomainDataExternal_, true, false, numThreads));
      transformXForward_.reset(new R2CTransform1DPlanesHost<T>(
          spaceDomainDataExternal_, freqDomainXY_, false, true, numThreads));
    } else {
      zStickSymmetry_.reset(new Symmetry());
      planeSymmetry_.reset(new Symmetry());

      auto spaceDomainData =
          create_3d_view(array1, 0, param->dim_z(), param->dim_y(), param->dim_x_freq());
      spaceDomainDataExternal_ =
          create_new_type_3d_view<T>(array1, param->dim_z(), param->dim_y(), 2 * param->dim_x());
      transformXBackward_.reset(new Transform1DPlanesHost<T>(freqDomainXY_, spaceDomainData, true,
                                                             false, FFTW_BACKWARD, numThreads));

      transformXForward_.reset(new Transform1DPlanesHost<T>(spaceDomainData, freqDomainXY_, false,
                                                            true, FFTW_FORWARD, numThreads));
    }
  }
}

#ifdef SPFFT_MPI

template <typename T>
ExecutionHost<T>::ExecutionHost(MPICommunicatorHandle comm, const SpfftExchangeType exchangeType,
                                const int numThreads, std::shared_ptr<Parameters> param,
                                HostArray<std::complex<T>>& array1,
                                HostArray<std::complex<T>>& array2)
    : numThreads_(numThreads),
      scalingFactor_(static_cast<T>(
          1.0 / static_cast<double>(param->dim_x() * param->dim_y() * param->dim_z()))),
      zStickSymmetry_(new Symmetry()),
      planeSymmetry_(new Symmetry()) {
  HOST_TIMING_SCOPED("Execution init");
  const SizeType numLocalZSticks = param->num_z_sticks(comm.rank());
  const SizeType numLocalXYPlanes = param->num_xy_planes(comm.rank());

  // get unique x indices to only compute non-zero y-transforms
  std::set<SizeType> uniqueXIndices;
  for (SizeType r = 0; r < comm.size(); ++r) {
    for (const auto& xyIndex : param->z_stick_xy_indices(r)) {
      uniqueXIndices.emplace(static_cast<SizeType>(xyIndex / param->dim_y()));
    }
  }

  auto freqDomainZ3D = create_3d_view(array1, 0, 1, numLocalZSticks, param->dim_z());
  freqDomainData_ = create_2d_view(freqDomainZ3D, 0, numLocalZSticks, param->dim_z());

  freqDomainXY_ = create_3d_view(array2, 0, numLocalXYPlanes, param->dim_x_freq(), param->dim_y());

  auto& spaceDomainArray = array1;
  // create external view with
  if (param->transform_type() == SPFFT_TRANS_R2C) {
    spaceDomainDataExternal_ = create_new_type_3d_view<T>(spaceDomainArray, numLocalXYPlanes,
                                                          param->dim_y(), param->dim_x());
  } else {
    spaceDomainDataExternal_ = create_new_type_3d_view<T>(spaceDomainArray, numLocalXYPlanes,
                                                          param->dim_y(), 2 * param->dim_x());
  }

  if (param->local_value_indices().size() > 0) {
    compression_.reset(new CompressionHost(param));
  }

  if (numLocalZSticks > 0) {
    // apply hermitian symmetry for x=0, y=0 stick
    if (param->transform_type() == SPFFT_TRANS_R2C &&
        param->zero_zero_stick_index() < freqDomainData_.dim_outer()) {
      zStickSymmetry_.reset(new StickSymmetryHost<T>(
          HostArrayView1D<std::complex<T>>(&freqDomainData_(param->zero_zero_stick_index(), 0),
                                           freqDomainData_.dim_inner(), freqDomainData_.pinned())));
    }
    transformZForward_ = std::unique_ptr<TransformHost<T>>(new Transform1DPlanesHost<T>(
        freqDomainZ3D, freqDomainZ3D, false, false, FFTW_FORWARD, numThreads));
    transformZBackward_ = std::unique_ptr<TransformHost<T>>(new Transform1DPlanesHost<T>(
        freqDomainZ3D, freqDomainZ3D, false, false, FFTW_BACKWARD, numThreads));
  }

  if (numLocalXYPlanes > 0) {
    transformYBackward_.reset(new Transform1DVerticalHost<T>(freqDomainXY_, freqDomainXY_, false,
                                                             false, FFTW_BACKWARD, uniqueXIndices));
    transformYForward_.reset(new Transform1DVerticalHost<T>(freqDomainXY_, freqDomainXY_, false,
                                                            false, FFTW_FORWARD, uniqueXIndices));

    if (param->transform_type() == SPFFT_TRANS_R2C) {
      transformXBackward_.reset(new C2RTransform1DPlanesHost<T>(
          freqDomainXY_, spaceDomainDataExternal_, true, false, numThreads));
      transformXForward_.reset(new R2CTransform1DPlanesHost<T>(
          spaceDomainDataExternal_, freqDomainXY_, false, true, numThreads));

      planeSymmetry_.reset(new PlaneSymmetryHost<T>(freqDomainXY_));

    } else {
      auto spaceDomainData =
          create_3d_view(spaceDomainArray, 0, numLocalXYPlanes, param->dim_y(), param->dim_x());
      transformXBackward_.reset(new Transform1DPlanesHost<T>(freqDomainXY_, spaceDomainData, true,
                                                             false, FFTW_BACKWARD, numThreads));
      transformXForward_.reset(new Transform1DPlanesHost<T>(spaceDomainData, freqDomainXY_, false,
                                                            true, FFTW_FORWARD, numThreads));
    }
  }

  switch (exchangeType) {
    case SpfftExchangeType::SPFFT_EXCH_UNBUFFERED: {
      transpose_.reset(
          new TransposeMPIUnbufferedHost<T>(param, comm, freqDomainXY_, freqDomainData_));
    } break;
    case SpfftExchangeType::SPFFT_EXCH_COMPACT_BUFFERED: {
      auto transposeBufferZ = create_1d_view(
          array2, 0, param->total_num_xy_planes() * param->num_z_sticks(comm.rank()));
      auto transposeBufferXY = create_1d_view(
          array1, 0, param->total_num_z_sticks() * param->num_xy_planes(comm.rank()));
      transpose_.reset(new TransposeMPICompactBufferedHost<T, T>(
          param, comm, freqDomainXY_, freqDomainData_, transposeBufferXY, transposeBufferZ));
    } break;
    case SpfftExchangeType::SPFFT_EXCH_COMPACT_BUFFERED_FLOAT: {
      auto transposeBufferZ = create_1d_view(
          array2, 0, param->total_num_xy_planes() * param->num_z_sticks(comm.rank()));
      auto transposeBufferXY = create_1d_view(
          array1, 0, param->total_num_z_sticks() * param->num_xy_planes(comm.rank()));
      transpose_.reset(new TransposeMPICompactBufferedHost<T, float>(
          param, comm, freqDomainXY_, freqDomainData_, transposeBufferXY, transposeBufferZ));
    } break;
    case SpfftExchangeType::SPFFT_EXCH_BUFFERED: {
      auto transposeBufferZ = create_1d_view(
          array2, 0, param->max_num_z_sticks() * param->max_num_xy_planes() * comm.size());
      auto transposeBufferXY = create_1d_view(
          array1, 0, param->max_num_z_sticks() * param->max_num_xy_planes() * comm.size());
      transpose_.reset(new TransposeMPIBufferedHost<T, T>(
          param, comm, freqDomainXY_, freqDomainData_, transposeBufferXY, transposeBufferZ));
    } break;
    case SpfftExchangeType::SPFFT_EXCH_BUFFERED_FLOAT: {
      auto transposeBufferZ = create_1d_view(
          array2, 0, param->max_num_z_sticks() * param->max_num_xy_planes() * comm.size());
      auto transposeBufferXY = create_1d_view(
          array1, 0, param->max_num_z_sticks() * param->max_num_xy_planes() * comm.size());
      transpose_.reset(new TransposeMPIBufferedHost<T, float>(
          param, comm, freqDomainXY_, freqDomainData_, transposeBufferXY, transposeBufferZ));
    } break;
    default:
      throw InvalidParameterError();
  }
}
#endif

template <typename T>
auto ExecutionHost<T>::forward_xy(const T* input) -> void {
  SPFFT_OMP_PRAGMA("omp parallel num_threads(numThreads_)") {
    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_START("x transform"); }
    if (transformXForward_) transformXForward_->execute(input, reinterpret_cast<T*>(freqDomainXY_.data()));
    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_STOP("x transform"); }

    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_START("y transform"); }
    if (transformYForward_) transformYForward_->execute();
    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_STOP("y transform"); }

    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_START("pack"); }
    if (transformYForward_) transpose_->pack_forward();
    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_STOP("pack"); }
  }
}

template <typename T>
auto ExecutionHost<T>::forward_exchange(const bool nonBlockingExchange) -> void {
  HOST_TIMING_SCOPED("exchange_start")
  // must be called outside omp parallel region (MPI restriction on thread id)
  transpose_->exchange_forward_start(nonBlockingExchange);
  // SPFFT_OMP_PRAGMA("omp barrier") // ensure exchange is done
}

template <typename T>
auto ExecutionHost<T>::forward_z(T* output, const SpfftScalingType scalingType) -> void {
  // must be called outside omp parallel region (MPI restriction on thread id)

  HOST_TIMING_START("exechange_fininalize");
  transpose_->exchange_forward_finalize();
  HOST_TIMING_STOP("exechange_fininalize");

  HOST_TIMING_STOP("exchange")
  SPFFT_OMP_PRAGMA("omp parallel num_threads(numThreads_)") {
    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_START("unpack"); }
    if (transformZForward_) transpose_->unpack_forward();
    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_STOP("unpack"); }

    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_START("z transform"); }
    if (transformZForward_) transformZForward_->execute();
    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_STOP("z transform"); }

    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_START("compression"); }
    if (compression_)
      compression_->compress(freqDomainData_, output,
                             scalingType == SpfftScalingType::SPFFT_FULL_SCALING, scalingFactor_);
    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_STOP("compression"); }
  }
}

template <typename T>
auto ExecutionHost<T>::backward_z(const T* input) -> void {
  SPFFT_OMP_PRAGMA("omp parallel num_threads(numThreads_)") {
    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_START("compression"); }
    if (compression_) compression_->decompress(input, freqDomainData_);
    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_STOP("compression"); }

    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_START("z symmetrization"); }
    zStickSymmetry_->apply();
    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_STOP("z symmetrization"); }

    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_START("z transform"); }
    if (transformZBackward_) transformZBackward_->execute();
    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_STOP("z transform"); }

    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_START("pack"); }
    if (transformZBackward_) transpose_->pack_backward();
    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_STOP("pack"); }
  }
}

template <typename T>
auto ExecutionHost<T>::backward_exchange(const bool nonBlockingExchange) -> void {
  HOST_TIMING_SCOPED("exchange_start")
  // must be called outside omp parallel region (MPI restriction on thread id)
  transpose_->exchange_backward_start(nonBlockingExchange);
}

template <typename T>
auto ExecutionHost<T>::backward_xy(T* output) -> void {
  // must be called outside omp parallel region (MPI restriction on thread id)
  HOST_TIMING_START("exechange_fininalize");
  transpose_->exchange_forward_finalize();
  HOST_TIMING_STOP("exechange_fininalize");

  HOST_TIMING_STOP("exchange")
  SPFFT_OMP_PRAGMA("omp parallel num_threads(numThreads_)") {
    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_START("unpack"); }
    if (transformYBackward_) transpose_->unpack_backward();
    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_STOP("unpack"); }

    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_START("xy symmetrization"); }
    planeSymmetry_->apply();
    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_STOP("xy symmetrization"); }

    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_START("y transform"); }
    if (transformYBackward_)
      transformYBackward_->execute();
    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_STOP("y transform"); }

    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_START("x transform"); }
    if (transformXBackward_)
      transformXBackward_->execute(reinterpret_cast<T*>(freqDomainXY_.data()), output);
    SPFFT_OMP_PRAGMA("omp master") { HOST_TIMING_STOP("x transform"); }
  }
}

template <typename T>
auto ExecutionHost<T>::space_domain_data() -> HostArrayView3D<T> {
  return spaceDomainDataExternal_;
}

// instatiate templates for float and double
template class ExecutionHost<double>;

#ifdef SPFFT_SINGLE_PRECISION
template class ExecutionHost<float>;
#endif

}  // namespace spfft
