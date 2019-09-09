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
#ifndef SPFFT_COMPRESSION_GPU_HPP
#define SPFFT_COMPRESSION_GPU_HPP

#include <complex>
#include <cstring>
#include <memory>
#include <vector>
#include "compression/gpu_kernels/compression_kernels.hpp"
#include "compression/indices.hpp"
#include "gpu_util/gpu_fft_api.hpp"
#include "gpu_util/gpu_stream_handle.hpp"
#include "gpu_util/gpu_transfer.hpp"
#include "memory/array_view_utility.hpp"
#include "memory/gpu_array.hpp"
#include "memory/gpu_array_view.hpp"
#include "parameters/parameters.hpp"
#include "spfft/config.h"
#include "util/common_types.hpp"
#include "util/type_check.hpp"

namespace spfft {

// Handles packing and unpacking of sparse frequency values for single or double precision on GPU
class CompressionGPU {
public:
  CompressionGPU(const std::shared_ptr<Parameters>& param)
      : indicesGPU_(
            param->local_value_indices().size()) {  // stream MUST synchronize with default stream
    copy_to_gpu(param->local_value_indices(), indicesGPU_);
  }

  // Pack values into output buffer
  template <typename T>
  auto compress(const GPUStreamHandle& stream,
                const GPUArrayView2D<typename gpu::fft::ComplexType<T>::type> input, T* output,
                const bool useScaling, const T scalingFactor = 1.0) -> void {
    static_assert(IsFloatOrDouble<T>::value, "Type T must be float or double");
    compress_gpu(stream.get(), create_1d_view(indicesGPU_, 0, indicesGPU_.size()), input, output,
                 useScaling, scalingFactor);
  }

  // Unpack values into z-stick collection
  template <typename T>
  auto decompress(const GPUStreamHandle& stream, const T* input,
                  GPUArrayView2D<typename gpu::fft::ComplexType<T>::type> output) -> void {
    static_assert(IsFloatOrDouble<T>::value, "Type T must be float or double");
    gpu::check_status(gpu::memset_async(
        static_cast<void*>(output.data()), 0,
        output.size() * sizeof(typename decltype(output)::ValueType), stream.get()));
    decompress_gpu(stream.get(), create_1d_view(indicesGPU_, 0, indicesGPU_.size()), input, output);
  }

private:
  GPUArray<int> indicesGPU_;
};
}  // namespace spfft

#endif
