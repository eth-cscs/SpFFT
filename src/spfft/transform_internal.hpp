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
#ifndef SPFFT_TRANSFORM_INTERNAL_HPP
#define SPFFT_TRANSFORM_INTERNAL_HPP

#include <memory>
#include "execution/execution_host.hpp"
#include "parameters/parameters.hpp"
#include "spfft/config.h"
#include "spfft/grid_internal.hpp"
#include "spfft/types.h"
#include "util/common_types.hpp"

#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
#include "compression/compression_gpu.hpp"
#include "execution/execution_gpu.hpp"
#endif

namespace spfft {
template <typename T>
class TransformInternal {
public:
  TransformInternal(SpfftProcessingUnitType executionUnit, std::shared_ptr<GridInternal<T>> grid,
                    std::shared_ptr<Parameters> param);

  auto clone() const -> TransformInternal<T>;

  inline auto type() const noexcept -> SpfftTransformType { return param_->transform_type(); }

  inline auto dim_x() const noexcept -> int { return param_->dim_x(); }

  inline auto dim_y() const noexcept -> int { return param_->dim_y(); }

  inline auto dim_z() const noexcept -> int { return param_->dim_z(); }

  inline auto num_local_xy_planes() const noexcept -> int { return param_->local_num_xy_planes(); }

  inline auto local_xy_plane_offset() const noexcept -> int {
    return param_->local_xy_plane_offset();
  }

  inline auto processing_unit() const noexcept -> SpfftProcessingUnitType { return executionUnit_; }

  inline auto device_id() const -> int { return grid_->device_id(); }

  inline auto num_threads() const -> int { return grid_->num_threads(); }

  inline auto num_local_elements() const -> int { return param_->local_num_elements(); }

  inline auto num_global_elements() const -> long long int { return param_->global_num_elements(); }

  inline auto global_size() const -> long long int { return param_->global_size(); }

  inline auto shared_grid(const TransformInternal<T>& other) const -> bool {
    return other.grid_ == grid_;
  }

  inline auto transform_type() const -> SpfftTransformType {
    return param_->transform_type();
  }

#ifdef SPFFT_MPI
  inline auto communicator() const -> MPI_Comm { return grid_->communicator().get(); }
#endif

  // full forward transform with blocking communication
  auto forward(const SpfftProcessingUnitType inputLocation, T* output, SpfftScalingType scaling)
      -> void;

  // transform in x and y
  auto forward_xy(const SpfftProcessingUnitType inputLocation) -> void;

  // start non-blocking exchange
  auto forward_exchange() -> void;

  // finalize exchange and transform z
  auto forward_z(T* output, SpfftScalingType scaling) -> void;

  // full backward transform with blocking communication
  auto backward(const T* input, const SpfftProcessingUnitType outputLocation) -> void;

  // transform in x and y
  auto backward_xy(const SpfftProcessingUnitType outputLocation) -> void;

  // start non-blocking exchange
  auto backward_exchange() -> void;

  // finalize exchange and transform z
  auto backward_z(const T* input) -> void;

  // must be called after step-wise transforms on GPUs
  auto synchronize() -> void;

  auto space_domain_data(SpfftProcessingUnitType location) -> T*;

private:
  SpfftProcessingUnitType executionUnit_;
  std::shared_ptr<Parameters> param_;
  std::shared_ptr<GridInternal<T>> grid_;

  std::unique_ptr<ExecutionHost<T>> execHost_;
#if (defined(SPFFT_CUDA) || defined(SPFFT_ROCM))
  std::unique_ptr<ExecutionGPU<T>> execGPU_;
#endif
};

} // namespace spfft

#endif

