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

#include <memory>
#include "compression/indices.hpp"
#include "execution/execution_host.hpp"
#include "parameters/parameters.hpp"
#include "spfft/config.h"
#include "spfft/exceptions.hpp"

#include "spfft/transform_internal.hpp"

#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
#include "gpu_util/gpu_device_guard.hpp"
#include "gpu_util/gpu_transfer.hpp"
#endif

namespace spfft {
template <typename T>
TransformInternal<T>::TransformInternal(SpfftProcessingUnitType executionUnit,
                                        std::shared_ptr<GridInternal<T>> grid,
                                        std::shared_ptr<Parameters> param)
    : executionUnit_(executionUnit), grid_(std::move(grid)), param_(std::move(param)) {
  // ----------------------
  // Input Check
  // ----------------------
  if (!grid_) {
    throw InvalidParameterError();
  }
  if (param_->local_num_xy_planes() > static_cast<SizeType>(grid_->max_num_local_xy_planes())) {
    throw InvalidParameterError();
  }
  if (grid_->local() && param_->dim_z() != param_->local_num_xy_planes()) {
    throw InvalidParameterError();
  }
  if (param_->local_num_z_sticks() > static_cast<SizeType>(grid_->max_num_local_z_columns())) {
    throw InvalidParameterError();
  }
  if (param_->dim_x() > static_cast<SizeType>(grid_->max_dim_x()) ||
      param_->dim_y() > static_cast<SizeType>(grid_->max_dim_y()) ||
      param_->dim_z() > static_cast<SizeType>(grid_->max_dim_z())) {
    throw InvalidParameterError();
  }
  if (!(executionUnit & grid_->processing_unit())) {
    // must match memory initialization parameters for grid
    throw InvalidParameterError();
  }
  if (executionUnit != SpfftProcessingUnitType::SPFFT_PU_HOST &&
      executionUnit != SpfftProcessingUnitType::SPFFT_PU_GPU) {
    // must be exclusively CPU or GPU
    throw InvalidParameterError();
  }
#ifdef SPFFT_MPI
  if (grid_->communicator().size() != param_->comm_size() ||
      grid_->communicator().rank() != param_->comm_rank()) {
    throw InternalError();
  }
#endif

  // create execution
  if (grid_->local()) {
    // ----------------------
    // Local
    // ----------------------
    if (executionUnit == SpfftProcessingUnitType::SPFFT_PU_HOST) {
      execHost_.reset(new ExecutionHost<T>(grid_->num_threads(), param_, grid_->array_host_1(),
                                           grid_->array_host_2()));
    } else {
      // GPU
#if (defined(SPFFT_CUDA) || defined(SPFFT_ROCM))
      execGPU_.reset(new ExecutionGPU<T>(grid_->num_threads(), param_, grid_->array_host_1(),
                                         grid_->array_host_2(), grid_->array_gpu_1(),
                                         grid_->array_gpu_2(), grid_->fft_work_buffer()));

#else
      throw GPUSupportError();
#endif
    }

  } else {
    // ----------------------
    // Distributed
    // ----------------------
#ifdef SPFFT_MPI
    if (executionUnit == SpfftProcessingUnitType::SPFFT_PU_HOST) {
      // CPU
      execHost_.reset(new ExecutionHost<T>(grid_->communicator(), grid_->exchange_type(),
                                           grid_->num_threads(), param_, grid_->array_host_1(),
                                           grid_->array_host_2()));
    } else {
      // GPU
#if (defined(SPFFT_CUDA) || defined(SPFFT_ROCM))
      // set device for current thread
      GPUDeviceGuard(grid_->device_id());

      execGPU_.reset(new ExecutionGPU<T>(grid_->communicator(), grid_->exchange_type(),
                                         grid_->num_threads(), param_, grid_->array_host_1(),
                                         grid_->array_host_2(), grid_->array_gpu_1(),
                                         grid_->array_gpu_2(), grid_->fft_work_buffer()));

#else   // GPU
      throw GPUSupportError();
#endif  // GPU
    }
#else   // MPI
    throw MPISupportError();
#endif  // MPI
  }
}

template <typename T>
auto TransformInternal<T>::forward(const SpfftProcessingUnitType inputLocation, T* output,
                                   SpfftScalingType scaling) -> void {
  HOST_TIMING_SCOPED("forward")
  if (executionUnit_ == SpfftProcessingUnitType::SPFFT_PU_HOST) {
    assert(execHost_);
    if (inputLocation != SpfftProcessingUnitType::SPFFT_PU_HOST) {
      throw InvalidParameterError();
    }
    execHost_->forward_xy();
    execHost_->forward_exchange(false);
    execHost_->forward_z(output, scaling);
  } else {
#if (defined(SPFFT_CUDA) || defined(SPFFT_ROCM))
    assert(execGPU_);
    // set device for current thread
    GPUDeviceGuard(grid_->device_id());

    execGPU_->forward_xy(inputLocation);
    execGPU_->forward_exchange(false);
    execGPU_->forward_z(output, scaling);
    execGPU_->synchronize();
#else
    throw GPUSupportError();
#endif
  }
}

template <typename T>
auto TransformInternal<T>::clone() const -> TransformInternal<T> {
  std::shared_ptr<GridInternal<T>> newGrid(new GridInternal<T>(*grid_));
  return TransformInternal<T>(executionUnit_, std::move(newGrid), param_);
}

template <typename T>
auto TransformInternal<T>::forward_xy(const SpfftProcessingUnitType inputLocation) -> void {
  if (executionUnit_ == SpfftProcessingUnitType::SPFFT_PU_HOST) {
    assert(execHost_);
    if (inputLocation != SpfftProcessingUnitType::SPFFT_PU_HOST) {
      throw InvalidParameterError();
    }
    execHost_->forward_xy();
  } else {
#if (defined(SPFFT_CUDA) || defined(SPFFT_ROCM))
    assert(execGPU_);
    // set device for current thread
    GPUDeviceGuard(grid_->device_id());

    execGPU_->forward_xy(inputLocation);
#else
    throw GPUSupportError();
#endif
  }
}

template <typename T>
auto TransformInternal<T>::forward_exchange() -> void {
  if (executionUnit_ == SpfftProcessingUnitType::SPFFT_PU_HOST) {
    assert(execHost_);
    execHost_->forward_exchange(true);
  } else {
#if (defined(SPFFT_CUDA) || defined(SPFFT_ROCM))
    assert(execGPU_);
    // set device for current thread
    GPUDeviceGuard(grid_->device_id());

    execGPU_->forward_exchange(true);
#else
    throw GPUSupportError();
#endif
  }
}

template <typename T>
auto TransformInternal<T>::forward_z(T* output, SpfftScalingType scaling) -> void {
  if (executionUnit_ == SpfftProcessingUnitType::SPFFT_PU_HOST) {
    assert(execHost_);
    execHost_->forward_z(output, scaling);
  } else {
#if (defined(SPFFT_CUDA) || defined(SPFFT_ROCM))
    assert(execGPU_);
    // set device for current thread
    GPUDeviceGuard(grid_->device_id());

    execGPU_->forward_z(output, scaling);
#else
    throw GPUSupportError();
#endif
  }
}

template <typename T>
auto TransformInternal<T>::backward(const T* input, const SpfftProcessingUnitType outputLocation)
    -> void {
  HOST_TIMING_SCOPED("backward")
  // check if input is can be accessed from gpu
  if (executionUnit_ == SpfftProcessingUnitType::SPFFT_PU_HOST) {
    assert(execHost_);
    if (outputLocation != SpfftProcessingUnitType::SPFFT_PU_HOST) {
      throw InvalidParameterError();
    }

    execHost_->backward_z(input);
    execHost_->backward_exchange(false);
    execHost_->backward_xy();
  } else {
#if (defined(SPFFT_CUDA) || defined(SPFFT_ROCM))
    // set device for current thread
    GPUDeviceGuard(grid_->device_id());
    execGPU_->backward_z(input);
    execGPU_->backward_exchange(false);
    execGPU_->backward_xy(outputLocation);
    execGPU_->synchronize();
#else
    throw GPUSupportError();
#endif
  }
}

template <typename T>
auto TransformInternal<T>::backward_z(const T* input) -> void {
  // check if input is can be accessed from gpu
  if (executionUnit_ == SpfftProcessingUnitType::SPFFT_PU_HOST) {
    assert(execHost_);

    execHost_->backward_z(input);
  } else {
#if (defined(SPFFT_CUDA) || defined(SPFFT_ROCM))
    // set device for current thread
    GPUDeviceGuard(grid_->device_id());
    execGPU_->backward_z(input);
#else
    throw GPUSupportError();
#endif
  }
}

template <typename T>
auto TransformInternal<T>::backward_exchange() -> void {
  // check if input is can be accessed from gpu
  if (executionUnit_ == SpfftProcessingUnitType::SPFFT_PU_HOST) {
    assert(execHost_);

    execHost_->backward_exchange(true);
  } else {
#if (defined(SPFFT_CUDA) || defined(SPFFT_ROCM))
    // set device for current thread
    GPUDeviceGuard(grid_->device_id());
    execGPU_->backward_exchange(true);
#else
    throw GPUSupportError();
#endif
  }
}

template <typename T>
auto TransformInternal<T>::backward_xy(const SpfftProcessingUnitType outputLocation) -> void {
  // check if input is can be accessed from gpu
  if (executionUnit_ == SpfftProcessingUnitType::SPFFT_PU_HOST) {
    assert(execHost_);
    if (outputLocation != SpfftProcessingUnitType::SPFFT_PU_HOST) {
      throw InvalidParameterError();
    }

    execHost_->backward_xy();
  } else {
#if (defined(SPFFT_CUDA) || defined(SPFFT_ROCM))
    // set device for current thread
    GPUDeviceGuard(grid_->device_id());
    execGPU_->backward_xy(outputLocation);
#else
    throw GPUSupportError();
#endif
  }
}

template <typename T>
auto TransformInternal<T>::space_domain_data(SpfftProcessingUnitType location) -> T* {
#if (defined(SPFFT_CUDA) || defined(SPFFT_ROCM))
  if (executionUnit_ == SpfftProcessingUnitType::SPFFT_PU_GPU) {
    // GPU
    if (location == SpfftProcessingUnitType::SPFFT_PU_GPU) {
      return execGPU_->space_domain_data_gpu().data();
    } else {
      return execGPU_->space_domain_data_host().data();
    }
  }
#endif

  // CPU
  if (location != SpfftProcessingUnitType::SPFFT_PU_HOST) throw InvalidParameterError();
  return execHost_->space_domain_data().data();
}

template <typename T>
auto TransformInternal<T>::synchronize() -> void {
#if (defined(SPFFT_CUDA) || defined(SPFFT_ROCM))
  if (execGPU_) execGPU_->synchronize();
#endif
}

// instatiate templates for float and double
template class TransformInternal<double>;
#ifdef SPFFT_SINGLE_PRECISION
template class TransformInternal<float>;
#endif

}  // namespace spfft
