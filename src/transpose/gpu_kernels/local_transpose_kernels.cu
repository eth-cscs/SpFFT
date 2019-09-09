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
#include "spfft/config.h"

namespace spfft {

// ------------------
// Backward
// ------------------

#ifdef SPFFT_CUDA
// kernel optimized for NVIDIA

// Places data from z-sticks into a full 3d grid
template <typename T>
__global__ static void transpose_backward_kernel(const GPUArrayConstView1D<int> indices,
                                                 const GPUArrayConstView2D<T> freqZData,
                                                 GPUArrayView2D<T> spaceDomainFlat) {
  // const int z = threadIdx.x + blockIdx.x * blockDim.x;
  const int stickIndex = threadIdx.x + blockIdx.x * blockDim.x;
  const auto stickXYIndex = indices(stickIndex);

  if (stickIndex < indices.size()) {
    for (int z = blockIdx.y; z < freqZData.dim_inner(); z += gridDim.y) {
      spaceDomainFlat(z, stickXYIndex) = freqZData(stickIndex, z);
    }
  }
}

auto local_transpose_backward(
    const gpu::StreamType stream, const GPUArrayView1D<int> indices,
    const GPUArrayView2D<typename gpu::fft::ComplexType<double>::type>& freqZData,
    GPUArrayView3D<typename gpu::fft::ComplexType<double>::type> spaceDomain) -> void {
  assert(indices.size() == freqZData.dim_outer());
  assert(indices.size() <= spaceDomain.dim_inner() * spaceDomain.dim_mid());
  assert(spaceDomain.dim_outer() == freqZData.dim_inner());
  const dim3 threadBlock(128);
  const dim3 threadGrid((freqZData.dim_outer() + threadBlock.x - 1) / threadBlock.x,
                        std::min(freqZData.dim_inner(), 2160));
  launch_kernel(transpose_backward_kernel<typename gpu::fft::ComplexType<double>::type>, threadGrid,
                threadBlock, 0, stream, indices, freqZData,
                GPUArrayView2D<typename gpu::fft::ComplexType<double>::type>(
                    spaceDomain.data(), spaceDomain.dim_outer(),
                    spaceDomain.dim_mid() * spaceDomain.dim_inner(), spaceDomain.device_id()));
}

auto local_transpose_backward(
    const gpu::StreamType stream, const GPUArrayView1D<int> indices,
    const GPUArrayView2D<typename gpu::fft::ComplexType<float>::type>& freqZData,
    GPUArrayView3D<typename gpu::fft::ComplexType<float>::type> spaceDomain) -> void {
  assert(indices.size() == freqZData.dim_outer());
  assert(indices.size() <= spaceDomain.dim_inner() * spaceDomain.dim_mid());
  assert(spaceDomain.dim_outer() == freqZData.dim_inner());
  const dim3 threadBlock(128);
  const dim3 threadGrid((freqZData.dim_outer() + threadBlock.x - 1) / threadBlock.x,
                        std::min(freqZData.dim_inner(), 2160));
  launch_kernel(transpose_backward_kernel<typename gpu::fft::ComplexType<float>::type>, threadGrid,
                threadBlock, 0, stream, indices, freqZData,
                GPUArrayView2D<typename gpu::fft::ComplexType<float>::type>(
                    spaceDomain.data(), spaceDomain.dim_outer(),
                    spaceDomain.dim_mid() * spaceDomain.dim_inner(), spaceDomain.device_id()));
}

#else
// kernel optimized for AMD
// (ideal memory access pattern is different)

template <typename T>
__global__ static void transpose_backward_kernel(const GPUArrayConstView1D<int> indices,
                                                 const GPUArrayConstView2D<T> freqZData,
                                                 GPUArrayView2D<T> spaceDomainFlat) {
  const int z = threadIdx.x + blockIdx.x * blockDim.x;

  if (z < freqZData.dim_inner()) {
    for (int stickIndex = blockIdx.y; stickIndex < indices.size(); stickIndex += gridDim.y) {
      const auto stickXYIndex = indices(stickIndex);
      spaceDomainFlat(z, stickXYIndex) = freqZData(stickIndex, z);
    }
  }
}

auto local_transpose_backward(
    const gpu::StreamType stream, const GPUArrayView1D<int> indices,
    const GPUArrayView2D<typename gpu::fft::ComplexType<double>::type>& freqZData,
    GPUArrayView3D<typename gpu::fft::ComplexType<double>::type> spaceDomain) -> void {
  assert(indices.size() == freqZData.dim_outer());
  assert(indices.size() <= spaceDomain.dim_inner() * spaceDomain.dim_mid());
  assert(spaceDomain.dim_outer() == freqZData.dim_inner());
  const dim3 threadBlock(128);
  const dim3 threadGrid((freqZData.dim_inner() + threadBlock.x - 1) / threadBlock.x,
                        std::min(freqZData.dim_outer(), 2160));
  launch_kernel(transpose_backward_kernel<typename gpu::fft::ComplexType<double>::type>, threadGrid,
                threadBlock, 0, stream, indices, freqZData,
                GPUArrayView2D<typename gpu::fft::ComplexType<double>::type>(
                    spaceDomain.data(), spaceDomain.dim_outer(),
                    spaceDomain.dim_mid() * spaceDomain.dim_inner(), spaceDomain.device_id()));
}

auto local_transpose_backward(
    const gpu::StreamType stream, const GPUArrayView1D<int> indices,
    const GPUArrayView2D<typename gpu::fft::ComplexType<float>::type>& freqZData,
    GPUArrayView3D<typename gpu::fft::ComplexType<float>::type> spaceDomain) -> void {
  assert(indices.size() == freqZData.dim_outer());
  assert(indices.size() <= spaceDomain.dim_inner() * spaceDomain.dim_mid());
  assert(spaceDomain.dim_outer() == freqZData.dim_inner());
  const dim3 threadBlock(128);
  const dim3 threadGrid((freqZData.dim_inner() + threadBlock.x - 1) / threadBlock.x,
                        std::min(freqZData.dim_outer(), 2160));
  launch_kernel(transpose_backward_kernel<typename gpu::fft::ComplexType<float>::type>, threadGrid,
                threadBlock, 0, stream, indices, freqZData,
                GPUArrayView2D<typename gpu::fft::ComplexType<float>::type>(
                    spaceDomain.data(), spaceDomain.dim_outer(),
                    spaceDomain.dim_mid() * spaceDomain.dim_inner(), spaceDomain.device_id()));
}
#endif

// ------------------
// Forward
// ------------------

template <typename T>
__global__ static void transpose_forward_kernel(const GPUArrayConstView1D<int> indices,
                                                const GPUArrayConstView2D<T> spaceDomainFlat,
                                                GPUArrayView2D<T> freqZData) {
  const int z = threadIdx.x + blockIdx.x * blockDim.x;

  if (z < freqZData.dim_inner()) {
    for (int stickIndex = blockIdx.y; stickIndex < indices.size(); stickIndex += gridDim.y) {
      const auto stickXYIndex = indices(stickIndex);
      freqZData(stickIndex, z) = spaceDomainFlat(z, stickXYIndex);
    }
  }
}

auto local_transpose_forward(
    const gpu::StreamType stream, const GPUArrayView1D<int> indices,
    const GPUArrayView3D<typename gpu::fft::ComplexType<double>::type>& spaceDomain,
    GPUArrayView2D<typename gpu::fft::ComplexType<double>::type> freqZData) -> void {
  assert(indices.size() == freqZData.dim_outer());
  assert(indices.size() <= spaceDomain.dim_inner() * spaceDomain.dim_mid());
  assert(spaceDomain.dim_outer() == freqZData.dim_inner());
  const dim3 threadBlock(128);
  const dim3 threadGrid((freqZData.dim_inner() + threadBlock.x - 1) / threadBlock.x,
                        std::min(freqZData.dim_outer(), 2160));
  launch_kernel(transpose_forward_kernel<typename gpu::fft::ComplexType<double>::type>, threadGrid,
                threadBlock, 0, stream, indices,
                GPUArrayConstView2D<typename gpu::fft::ComplexType<double>::type>(
                    spaceDomain.data(), spaceDomain.dim_outer(),
                    spaceDomain.dim_mid() * spaceDomain.dim_inner(), spaceDomain.device_id()),
                freqZData);
}

auto local_transpose_forward(
    const gpu::StreamType stream, const GPUArrayView1D<int> indices,
    const GPUArrayView3D<typename gpu::fft::ComplexType<float>::type>& spaceDomain,
    GPUArrayView2D<typename gpu::fft::ComplexType<float>::type> freqZData) -> void {
  assert(indices.size() == freqZData.dim_outer());
  assert(indices.size() <= spaceDomain.dim_inner() * spaceDomain.dim_mid());
  assert(spaceDomain.dim_outer() == freqZData.dim_inner());
  const dim3 threadBlock(128);
  const dim3 threadGrid((freqZData.dim_inner() + threadBlock.x - 1) / threadBlock.x,
                        std::min(freqZData.dim_outer(), 2160));
  launch_kernel(transpose_forward_kernel<typename gpu::fft::ComplexType<float>::type>, threadGrid,
                threadBlock, 0, stream, indices,
                GPUArrayConstView2D<typename gpu::fft::ComplexType<float>::type>(
                    spaceDomain.data(), spaceDomain.dim_outer(),
                    spaceDomain.dim_mid() * spaceDomain.dim_inner(), spaceDomain.device_id()),
                freqZData);
}

}  // namespace spfft
