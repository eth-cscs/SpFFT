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
#include "gpu_util/gpu_fft_api.hpp"
#include "gpu_util/gpu_runtime.hpp"
#include "memory/array_view_utility.hpp"
#include "memory/gpu_array_const_view.hpp"
#include "memory/gpu_array_view.hpp"

namespace spfft {

template <typename T>
__global__ static void decompress_kernel(
    const GPUArrayConstView1D<int> indices, const T* input,
    GPUArrayView1D<typename gpu::fft::ComplexType<T>::type> output) {
  // const int stride = gridDim.x * blockDim.x;
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < indices.size();
       idx += gridDim.x * blockDim.x) {
    const int valueIdx = indices(idx);
    typename gpu::fft::ComplexType<T>::type value;
    value.x = input[2 * idx];
    value.y = input[2 * idx + 1];
    output(valueIdx) = value;
  }
}

auto decompress_gpu(const gpu::StreamType stream, const GPUArrayView1D<int>& indices,
                    const double* input,
                    GPUArrayView2D<typename gpu::fft::ComplexType<double>::type> output) -> void {
  assert(indices.size() <= output.size());
  const dim3 threadBlock(256);
  const dim3 threadGrid(
      std::min(static_cast<int>((indices.size() + threadBlock.x - 1) / threadBlock.x), 4320));
  // const dim3 threadGrid(indices.size() < 4 ? 1 : indices.size() / 4);
  launch_kernel(decompress_kernel<double>, threadGrid, threadBlock, 0, stream, indices, input,
                create_1d_view(output, 0, output.size()));
}

auto decompress_gpu(const gpu::StreamType stream, const GPUArrayView1D<int>& indices,
                    const float* input,
                    GPUArrayView2D<typename gpu::fft::ComplexType<float>::type> output) -> void {
  assert(indices.size() <= output.size());
  const dim3 threadBlock(256);
  const dim3 threadGrid(
      std::min(static_cast<int>((indices.size() + threadBlock.x - 1) / threadBlock.x), 4320));
  launch_kernel(decompress_kernel<float>, threadGrid, threadBlock, 0, stream, indices, input,
                create_1d_view(output, 0, output.size()));
}

template <typename T>
__global__ static void compress_kernel(
    const GPUArrayConstView1D<int> indices,
    GPUArrayConstView1D<typename gpu::fft::ComplexType<T>::type> input, T* output) {
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < indices.size();
       idx += gridDim.x * blockDim.x) {
    const int valueIdx = indices(idx);
    const auto value = input(valueIdx);
    output[2 * idx] = value.x;
    output[2 * idx + 1] = value.y;
  }
}

template <typename T>
__global__ static void compress_kernel_scaled(
    const GPUArrayConstView1D<int> indices,
    GPUArrayConstView1D<typename gpu::fft::ComplexType<T>::type> input, T* output,
    const T scalingFactor) {
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < indices.size();
       idx += gridDim.x * blockDim.x) {
    const int valueIdx = indices(idx);
    const auto value = input(valueIdx);
    output[2 * idx] = scalingFactor * value.x;
    output[2 * idx + 1] = scalingFactor * value.y;
  }
}

auto compress_gpu(const gpu::StreamType stream, const GPUArrayView1D<int>& indices,
                  GPUArrayView2D<typename gpu::fft::ComplexType<double>::type> input,
                  double* output, const bool useScaling, const double scalingFactor) -> void {
  const dim3 threadBlock(256);
  const dim3 threadGrid(
      std::min(static_cast<int>((indices.size() + threadBlock.x - 1) / threadBlock.x), 4320));

  if (useScaling) {
    launch_kernel(compress_kernel_scaled<double>, threadGrid, threadBlock, 0, stream, indices,
                  create_1d_view(input, 0, input.size()), output, scalingFactor);
  } else {
    launch_kernel(compress_kernel<double>, threadGrid, threadBlock, 0, stream, indices,
                  create_1d_view(input, 0, input.size()), output);
  }
}

auto compress_gpu(const gpu::StreamType stream, const GPUArrayView1D<int>& indices,
                  GPUArrayView2D<typename gpu::fft::ComplexType<float>::type> input, float* output,
                  const bool useScaling, const float scalingFactor) -> void {
  const dim3 threadBlock(256);
  const dim3 threadGrid(
      std::min(static_cast<int>((indices.size() + threadBlock.x - 1) / threadBlock.x), 4320));
  if (useScaling) {
    launch_kernel(compress_kernel_scaled<float>, threadGrid, threadBlock, 0, stream, indices,
                  create_1d_view(input, 0, input.size()), output, scalingFactor);
  } else {
    launch_kernel(compress_kernel<float>, threadGrid, threadBlock, 0, stream, indices,
                  create_1d_view(input, 0, input.size()), output);
  }
}
}  // namespace spfft
