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
#ifndef SPFFT_TRANSFORM_2D_GPU_HPP
#define SPFFT_TRANSFORM_2D_GPU_HPP

#include <cassert>
#include <complex>
#include <cstddef>
#include <memory>
#include "fft/transform_interface.hpp"
#include "gpu_util/gpu_fft_api.hpp"
#include "gpu_util/gpu_runtime_api.hpp"
#include "gpu_util/gpu_stream_handle.hpp"
#include "memory/gpu_array.hpp"
#include "memory/gpu_array_view.hpp"
#include "spfft/config.h"
#include "util/common_types.hpp"

namespace spfft {

template <typename T>
class Transform2DGPU : public TransformGPU {
public:
  using ValueType = T;
  using ComplexType = gpu::fft::ComplexType<T>;

  Transform2DGPU(GPUArrayView3D<typename gpu::fft::ComplexType<T>::type>& data,
                 GPUStreamHandle stream, std::shared_ptr<GPUArray<char>> workBuffer)
      : stream_(std::move(stream)), workBuffer_(std::move(workBuffer)), dataPtr_(data.data()) {
    assert(workBuffer_);

    std::size_t worksize = 0;

    int rank = 2;
    int n[2] = {data.dim_mid(), data.dim_inner()};
    int nembed[2] = {data.dim_mid(), data.dim_inner()};
    int stride = 1;
    int dist = data.dim_inner() * data.dim_mid();
    int batch = data.dim_outer();

    // create plan
    gpu::fft::check_result(gpu::fft::create(&plan_));
    gpu::fft::check_result(gpu::fft::set_auto_allocation(plan_, 0));
    gpu::fft::check_result(gpu::fft::make_plan_many(
        plan_, rank, n, nembed, stride, dist, nembed, stride, dist,
        gpu::fft::TransformType::ComplexToComplex<T>::value, batch, &worksize));

    // set stream
    gpu::fft::check_result(gpu::fft::set_stream(plan_, stream_.get()));

    // resize work buffer if necessary
    if (workBuffer_->size() < worksize) {
      *workBuffer_ = GPUArray<char>(worksize);
    }
  }

  Transform2DGPU(const Transform2DGPU& transform) = delete;

  Transform2DGPU(Transform2DGPU&& transform) noexcept
      : stream_(std::move(transform.stream_)),
        plan_(std::move(transform.plan_)),
        workBuffer_(std::move(transform.workBuffer_)),
        dataPtr_(transform.dataPtr_) {
    transform.plan_ = 0;
  }

  ~Transform2DGPU() {
    if (plan_) {
      gpu::fft::destroy(plan_);
    }
  }

  auto operator=(const Transform2DGPU& transform) -> Transform2DGPU& = delete;

  auto operator=(Transform2DGPU&& transform) noexcept -> Transform2DGPU& {
    if (plan_) {
      gpu::fft::destroy(plan_);
    }
    stream_ = std::move(transform.stream_);
    plan_ = std::move(transform.plan_);
    workBuffer_ = std::move(transform.workBuffer_);
    dataPtr_ = transform.dataPtr_;

    transform.plan_ = 0;
    return *this;
  }

  inline auto device_id() const noexcept -> int { return stream_.device_id(); }

  auto forward() -> void override {
    gpu::fft::check_result(gpu::fft::set_work_area(plan_, workBuffer_->data()));
    gpu::fft::check_result(
        gpu::fft::execute(plan_, dataPtr_, dataPtr_, gpu::fft::TransformDirection::Forward));
  }

  auto backward() -> void override {
    gpu::fft::check_result(gpu::fft::set_work_area(plan_, workBuffer_->data()));
    gpu::fft::check_result(
        gpu::fft::execute(plan_, dataPtr_, dataPtr_, gpu::fft::TransformDirection::Backward));
  }

private:
  GPUStreamHandle stream_;
  gpu::fft::HandleType plan_ = 0;
  std::shared_ptr<GPUArray<char>> workBuffer_;
  typename gpu::fft::ComplexType<T>::type* dataPtr_;
};
} // namespace spfft

#endif
