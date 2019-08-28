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
#ifndef SPFFT_TRANSPOSE_GPU_HPP
#define SPFFT_TRANSPOSE_GPU_HPP

#include <algorithm>
#include <cassert>
#include <complex>
#include <vector>
#include <memory>
#include "parameters/parameters.hpp"
#include "spfft/config.h"
#include "spfft/exceptions.hpp"
#include "transpose.hpp"
#include "util/common_types.hpp"
#include "util/type_check.hpp"

#include "gpu_util/gpu_fft_api.hpp"
#include "gpu_util/gpu_stream_handle.hpp"
#include "memory/array_view_utility.hpp"
#include "memory/gpu_array.hpp"
#include "memory/gpu_array_view.hpp"
#include "transpose/gpu_kernels/local_transpose_kernels.hpp"

namespace spfft {
// Transpose Z sticks, such that data is represented by xy planes, where the y-dimension is
// continous and vice versa
template <typename T>
class TransposeGPU : public Transpose {
  static_assert(IsFloatOrDouble<T>::value, "Type T must be float or double");
  using ValueType = T;
  using ComplexType = typename gpu::fft::ComplexType<ValueType>::type;

public:
  TransposeGPU(const std::shared_ptr<Parameters>& param, GPUStreamHandle stream,
               GPUArrayView3D<ComplexType> spaceDomainData,
               GPUArrayView2D<ComplexType> freqDomainData)
      : stream_(std::move(stream)),
        spaceDomainData_(spaceDomainData),
        freqDomainData_(freqDomainData), indices_(param->num_z_sticks(0)) {
    // single node only checks
    assert(spaceDomainData.dim_outer() == freqDomainData.dim_inner());

    // check data dimensions and parameters
    assert(param->dim_x_freq() == spaceDomainData.dim_inner());
    assert(param->dim_y() == spaceDomainData.dim_mid());
    assert(param->dim_z() == spaceDomainData.dim_outer());
    assert(param->dim_z() == freqDomainData.dim_inner());
    assert(param->num_z_sticks(0) == freqDomainData.dim_outer());

    // data must be disjoint
    assert(disjoint(spaceDomainData, freqDomainData));

    // copy xy indices
    const auto zStickXYIndices = param->z_stick_xy_indices(0);

    std::vector<int> transposedIndices;
    transposedIndices.reserve(zStickXYIndices.size());

    for(const auto& index : zStickXYIndices) {
      const int x = index / param->dim_y();
      const int y = index - x * param->dim_y();
      transposedIndices.emplace_back(y * param->dim_x_freq() + x);
    }

    copy_to_gpu(transposedIndices, indices_);
  }

  auto exchange_backward_start(const bool) -> void override {
    gpu::check_status(gpu::memset_async(
        static_cast<void*>(spaceDomainData_.data()), 0,
        spaceDomainData_.size() * sizeof(typename decltype(spaceDomainData_)::ValueType),
        stream_.get()));
    if (freqDomainData_.size() > 0 && spaceDomainData_.size() > 0) {
      local_transpose_backward(stream_.get(), create_1d_view(indices_, 0, indices_.size()),
                               freqDomainData_, spaceDomainData_);
    }
  }

  auto unpack_backward() -> void override {}

  auto exchange_forward_start(const bool) -> void override {
    if (freqDomainData_.size() > 0 && spaceDomainData_.size() > 0) {
      local_transpose_forward(stream_.get(), create_1d_view(indices_, 0, indices_.size()),
                              spaceDomainData_, freqDomainData_);
    }
  }

  auto unpack_forward() -> void override {}

private:
  GPUStreamHandle stream_;
  GPUArrayView3D<ComplexType> spaceDomainData_;
  GPUArrayView2D<ComplexType> freqDomainData_;
  GPUArray<int> indices_;
};
} // namespace spfft
#endif
