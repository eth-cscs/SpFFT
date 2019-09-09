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
#include <cassert>
#include "gpu_util/gpu_fft_api.hpp"
#include "gpu_util/gpu_runtime.hpp"
#include "memory/gpu_array_const_view.hpp"
#include "memory/gpu_array_view.hpp"

namespace spfft {

template <typename T>
__global__ static void symmetrize_plane_kernel(
    GPUArrayView3D<typename gpu::fft::ComplexType<T>::type> data, const int startIndex,
    const int numIndices) {
  assert(startIndex + numIndices <= data.dim_mid());
  int idxMid = threadIdx.x + blockIdx.x * blockDim.x;
  if (idxMid < numIndices) {
    idxMid += startIndex;
    auto value = data(blockIdx.y, idxMid, 0);
    if (value.x != T(0) || value.y != T(0)) {
      value.y = -value.y;
      data(blockIdx.y, data.dim_mid() - idxMid, 0) = value;
    }
  }
}

auto symmetrize_plane_gpu(const gpu::StreamType stream,
                          const GPUArrayView3D<typename gpu::fft::ComplexType<double>::type>& data)
    -> void {
  assert(data.size() > 2);
  {
    const int startIndex = 1;
    const int numIndices = data.dim_mid() / 2;
    const dim3 threadBlock(256);
    const dim3 threadGrid((numIndices + threadBlock.x - 1) / threadBlock.x, data.dim_outer());
    launch_kernel(symmetrize_plane_kernel<double>, threadGrid, threadBlock, 0, stream, data,
                  startIndex, numIndices);
  }
  {
    const int startIndex = data.dim_mid() / 2 + 1;
    const int numIndices = data.dim_mid() - startIndex;
    const dim3 threadBlock(256);
    const dim3 threadGrid((numIndices + threadBlock.x - 1) / threadBlock.x, data.dim_outer());
    launch_kernel(symmetrize_plane_kernel<double>, threadGrid, threadBlock, 0, stream, data,
                  startIndex, numIndices);
  }
}

auto symmetrize_plane_gpu(const gpu::StreamType stream,
                          const GPUArrayView3D<typename gpu::fft::ComplexType<float>::type>& data)
    -> void {
  assert(data.size() > 2);
  {
    const int startIndex = 1;
    const int numIndices = data.dim_mid() / 2;
    const dim3 threadBlock(256);
    const dim3 threadGrid((numIndices + threadBlock.x - 1) / threadBlock.x, data.dim_outer());
    launch_kernel(symmetrize_plane_kernel<float>, threadGrid, threadBlock, 0, stream, data,
                  startIndex, numIndices);
  }
  {
    const int startIndex = data.dim_mid() / 2 + 1;
    const int numIndices = data.dim_mid() - startIndex;
    const dim3 threadBlock(256);
    const dim3 threadGrid((numIndices + threadBlock.x - 1) / threadBlock.x, data.dim_outer());
    launch_kernel(symmetrize_plane_kernel<float>, threadGrid, threadBlock, 0, stream, data,
                  startIndex, numIndices);
  }
}

template <typename T>
__global__ static void symmetrize_stick_kernel(
    GPUArrayView1D<typename gpu::fft::ComplexType<T>::type> data, const int startIndex,
    const int numIndices) {
  assert(startIndex + numIndices <= data.size());
  int idxInner = threadIdx.x + blockIdx.x * blockDim.x;
  if (idxInner < numIndices) {
    idxInner += startIndex;
    auto value = data(idxInner);
    if (value.x != T(0) || value.y != T(0)) {
      value.y = -value.y;
      data(data.size() - idxInner) = value;
    }
  }
}

auto symmetrize_stick_gpu(const gpu::StreamType stream,
                          const GPUArrayView1D<typename gpu::fft::ComplexType<double>::type>& data)
    -> void {
  assert(data.size() > 2);
  {
    const int startIndex = 1;
    const int numIndices = data.size() / 2;
    const dim3 threadBlock(256);
    const dim3 threadGrid((numIndices + threadBlock.x - 1) / threadBlock.x);
    launch_kernel(symmetrize_stick_kernel<double>, threadGrid, threadBlock, 0, stream, data,
                  startIndex, numIndices);
  }
  {
    const int startIndex = data.size() / 2 + 1;
    const int numIndices = data.size() - startIndex;
    const dim3 threadBlock(256);
    const dim3 threadGrid((numIndices + threadBlock.x - 1) / threadBlock.x);
    launch_kernel(symmetrize_stick_kernel<double>, threadGrid, threadBlock, 0, stream, data,
                  startIndex, numIndices);
  }
}

auto symmetrize_stick_gpu(const gpu::StreamType stream,
                          const GPUArrayView1D<typename gpu::fft::ComplexType<float>::type>& data)
    -> void {
  assert(data.size() > 2);
  {
    const int startIndex = 1;
    const int numIndices = data.size() / 2;
    const dim3 threadBlock(256);
    const dim3 threadGrid((numIndices + threadBlock.x - 1) / threadBlock.x);
    launch_kernel(symmetrize_stick_kernel<float>, threadGrid, threadBlock, 0, stream, data,
                  startIndex, numIndices);
  }
  {
    const int startIndex = data.size() / 2 + 1;
    const int numIndices = data.size() - startIndex;
    const dim3 threadBlock(256);
    const dim3 threadGrid((numIndices + threadBlock.x - 1) / threadBlock.x);
    launch_kernel(symmetrize_stick_kernel<float>, threadGrid, threadBlock, 0, stream, data,
                  startIndex, numIndices);
  }
}

}  // namespace spfft
