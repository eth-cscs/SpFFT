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
#include "gpu_util/complex_conversion.cuh"
#include "gpu_util/gpu_fft_api.hpp"
#include "gpu_util/gpu_kernel_parameter.hpp"
#include "gpu_util/gpu_runtime.hpp"
#include "memory/array_view_utility.hpp"
#include "memory/gpu_array_const_view.hpp"

namespace spfft {

template <typename DATA_TYPE, typename BUFFER_TYPE>
__global__ static void compact_buffered_pack_backward_kernel(
    const GPUArrayConstView1D<int> numXYPlanes, const GPUArrayConstView1D<int> xyPlaneOffsets,
    const GPUArrayConstView2D<DATA_TYPE> freqZData, GPUArrayView1D<BUFFER_TYPE> buffer) {
  const int xyPlaneIndex = threadIdx.x + blockIdx.x * blockDim.x;
  int bufferOffset = 0;
  for (int r = 0; r < numXYPlanes.size(); ++r) {
    const int numCurrentXYPlanes = numXYPlanes(r);
    if (xyPlaneIndex < numCurrentXYPlanes) {
      for (int zStickIndex = blockIdx.y; zStickIndex < freqZData.dim_outer();
           zStickIndex += gridDim.y) {
        buffer(bufferOffset + zStickIndex * numCurrentXYPlanes + xyPlaneIndex) =
            ConvertComplex<BUFFER_TYPE, DATA_TYPE>::apply(
                freqZData(zStickIndex, xyPlaneIndex + xyPlaneOffsets(r)));
      }
    }
    bufferOffset += numCurrentXYPlanes * freqZData.dim_outer();
  }
}

template <typename DATA_TYPE, typename BUFFER_TYPE>
static auto compact_buffered_pack_backward_launch(const gpu::StreamType stream,
                                                  const int maxNumXYPlanes,
                                                  const GPUArrayView1D<int> numXYPlanes,
                                                  const GPUArrayView1D<int>& xyPlaneOffsets,
                                                  const GPUArrayView2D<DATA_TYPE>& freqZData,
                                                  GPUArrayView1D<BUFFER_TYPE> buffer) -> void {
  assert(xyPlaneOffsets.size() == numXYPlanes.size());
  const dim3 threadBlock(gpu::BlockSizeSmall);
  const dim3 threadGrid((maxNumXYPlanes + threadBlock.x - 1) / threadBlock.x,
                        std::min(freqZData.dim_outer(), gpu::GridSizeMedium));
  launch_kernel(compact_buffered_pack_backward_kernel<DATA_TYPE, BUFFER_TYPE>, threadGrid,
                threadBlock, 0, stream, numXYPlanes, xyPlaneOffsets, freqZData, buffer);
}

auto compact_buffered_pack_backward(
    const gpu::StreamType stream, const int maxNumXYPlanes, const GPUArrayView1D<int> numXYPlanes,
    const GPUArrayView1D<int>& xyPlaneOffsets,
    const GPUArrayView2D<typename gpu::fft::ComplexType<double>::type>& freqZData,
    GPUArrayView1D<typename gpu::fft::ComplexType<double>::type> buffer) -> void {
  compact_buffered_pack_backward_launch(stream, maxNumXYPlanes, numXYPlanes, xyPlaneOffsets,
                                        freqZData, buffer);
}

auto compact_buffered_pack_backward(
    const gpu::StreamType stream, const int maxNumXYPlanes, const GPUArrayView1D<int> numXYPlanes,
    const GPUArrayView1D<int>& xyPlaneOffsets,
    const GPUArrayView2D<typename gpu::fft::ComplexType<float>::type>& freqZData,
    GPUArrayView1D<typename gpu::fft::ComplexType<float>::type> buffer) -> void {
  compact_buffered_pack_backward_launch(stream, maxNumXYPlanes, numXYPlanes, xyPlaneOffsets,
                                        freqZData, buffer);
}

auto compact_buffered_pack_backward(
    const gpu::StreamType stream, const int maxNumXYPlanes, const GPUArrayView1D<int> numXYPlanes,
    const GPUArrayView1D<int>& xyPlaneOffsets,
    const GPUArrayView2D<typename gpu::fft::ComplexType<double>::type>& freqZData,
    GPUArrayView1D<typename gpu::fft::ComplexType<float>::type> buffer) -> void {
  compact_buffered_pack_backward_launch(stream, maxNumXYPlanes, numXYPlanes, xyPlaneOffsets,
                                        freqZData, buffer);
}

template <typename DATA_TYPE, typename BUFFER_TYPE>
__global__ static void compact_buffered_unpack_backward_kernel(
    const int maxNumZSticks, const GPUArrayConstView1D<int> numZSticks,
    const GPUArrayConstView1D<int> indices, const GPUArrayConstView1D<BUFFER_TYPE> buffer,
    GPUArrayView2D<DATA_TYPE> freqXYData) {
  const int xyPlaneIndex = threadIdx.x + blockIdx.x * blockDim.x;
  int bufferOffset = 0;
  if (xyPlaneIndex < freqXYData.dim_outer()) {
    for (int r = 0; r < numZSticks.size(); ++r) {
      const int numCurrentZSticks = numZSticks(r);
      for (int zStickIndex = blockIdx.y; zStickIndex < numCurrentZSticks;
           zStickIndex += gridDim.y) {
        const int currentIndex = indices(r * maxNumZSticks + zStickIndex);
        freqXYData(xyPlaneIndex, currentIndex) = ConvertComplex<DATA_TYPE, BUFFER_TYPE>::apply(
            buffer(bufferOffset + zStickIndex * freqXYData.dim_outer() + xyPlaneIndex));
      }
      bufferOffset += numCurrentZSticks * freqXYData.dim_outer();
    }
  }
}

template <typename DATA_TYPE, typename BUFFER_TYPE>
static auto compact_buffered_unpack_backward_launch(const gpu::StreamType stream,
                                                    const int maxNumZSticks,
                                                    const GPUArrayView1D<int>& numZSticks,
                                                    const GPUArrayView1D<int>& indices,
                                                    const GPUArrayView1D<BUFFER_TYPE>& buffer,
                                                    GPUArrayView3D<DATA_TYPE> freqXYData) -> void {
  const dim3 threadBlock(gpu::BlockSizeSmall);
  const dim3 threadGrid((freqXYData.dim_outer() + threadBlock.x - 1) / threadBlock.x,
                        std::min(maxNumZSticks, gpu::GridSizeMedium));
  launch_kernel(compact_buffered_unpack_backward_kernel<DATA_TYPE, BUFFER_TYPE>, threadGrid,
                threadBlock, 0, stream, maxNumZSticks, numZSticks, indices, buffer,
                GPUArrayView2D<DATA_TYPE>(freqXYData.data(), freqXYData.dim_outer(),
                                          freqXYData.dim_mid() * freqXYData.dim_inner(),
                                          freqXYData.device_id()));
}

auto compact_buffered_unpack_backward(
    const gpu::StreamType stream, const int maxNumZSticks, const GPUArrayView1D<int>& numZSticks,
    const GPUArrayView1D<int>& indices,
    const GPUArrayView1D<typename gpu::fft::ComplexType<double>::type>& buffer,
    GPUArrayView3D<typename gpu::fft::ComplexType<double>::type> freqXYData) -> void {
  compact_buffered_unpack_backward_launch(stream, maxNumZSticks, numZSticks, indices, buffer,
                                          freqXYData);
}

auto compact_buffered_unpack_backward(
    const gpu::StreamType stream, const int maxNumZSticks, const GPUArrayView1D<int>& numZSticks,
    const GPUArrayView1D<int>& indices,
    const GPUArrayView1D<typename gpu::fft::ComplexType<float>::type>& buffer,
    GPUArrayView3D<typename gpu::fft::ComplexType<float>::type> freqXYData) -> void {
  compact_buffered_unpack_backward_launch(stream, maxNumZSticks, numZSticks, indices, buffer,
                                          freqXYData);
}

auto compact_buffered_unpack_backward(
    const gpu::StreamType stream, const int maxNumZSticks, const GPUArrayView1D<int>& numZSticks,
    const GPUArrayView1D<int>& indices,
    const GPUArrayView1D<typename gpu::fft::ComplexType<float>::type>& buffer,
    GPUArrayView3D<typename gpu::fft::ComplexType<double>::type> freqXYData) -> void {
  compact_buffered_unpack_backward_launch(stream, maxNumZSticks, numZSticks, indices, buffer,
                                          freqXYData);
}

template <typename DATA_TYPE, typename BUFFER_TYPE>
__global__ static void compact_buffered_unpack_forward_kernel(
    const GPUArrayConstView1D<int> numXYPlanes, const GPUArrayConstView1D<int> xyPlaneOffsets,
    const GPUArrayConstView1D<BUFFER_TYPE> buffer, GPUArrayView2D<DATA_TYPE> freqZData) {
  const int xyPlaneIndex = threadIdx.x + blockIdx.x * blockDim.x;
  int bufferOffset = 0;
  for (int r = 0; r < numXYPlanes.size(); ++r) {
    const int numCurrentXYPlanes = numXYPlanes(r);
    if (xyPlaneIndex < numCurrentXYPlanes) {
      for (int zStickIndex = blockIdx.y; zStickIndex < freqZData.dim_outer();
           zStickIndex += gridDim.y) {
        freqZData(zStickIndex, xyPlaneIndex + xyPlaneOffsets(r)) =
            ConvertComplex<DATA_TYPE, BUFFER_TYPE>::apply(
                buffer(bufferOffset + zStickIndex * numCurrentXYPlanes + xyPlaneIndex));
      }
    }
    bufferOffset += numCurrentXYPlanes * freqZData.dim_outer();
  }
}

template <typename DATA_TYPE, typename BUFFER_TYPE>
static auto compact_buffered_unpack_forward_launch(const gpu::StreamType stream,
                                                   const int maxNumXYPlanes,
                                                   const GPUArrayView1D<int> numXYPlanes,
                                                   const GPUArrayView1D<int>& xyPlaneOffsets,
                                                   const GPUArrayView1D<BUFFER_TYPE>& buffer,
                                                   GPUArrayView2D<DATA_TYPE> freqZData) -> void {
  assert(xyPlaneOffsets.size() == numXYPlanes.size());
  const dim3 threadBlock(gpu::BlockSizeSmall);
  const dim3 threadGrid((maxNumXYPlanes + threadBlock.x - 1) / threadBlock.x,
                        std::min(freqZData.dim_outer(), gpu::GridSizeMedium));
  launch_kernel(compact_buffered_unpack_forward_kernel<DATA_TYPE, BUFFER_TYPE>, threadGrid,
                threadBlock, 0, stream, numXYPlanes, xyPlaneOffsets, buffer, freqZData);
}

auto compact_buffered_unpack_forward(
    const gpu::StreamType stream, const int maxNumXYPlanes, const GPUArrayView1D<int> numXYPlanes,
    const GPUArrayView1D<int>& xyPlaneOffsets,
    const GPUArrayView1D<typename gpu::fft::ComplexType<double>::type>& buffer,
    GPUArrayView2D<typename gpu::fft::ComplexType<double>::type> freqZData) -> void {
  compact_buffered_unpack_forward_launch(stream, maxNumXYPlanes, numXYPlanes, xyPlaneOffsets,
                                         buffer, freqZData);
}

auto compact_buffered_unpack_forward(
    const gpu::StreamType stream, const int maxNumXYPlanes, const GPUArrayView1D<int> numXYPlanes,
    const GPUArrayView1D<int>& xyPlaneOffsets,
    const GPUArrayView1D<typename gpu::fft::ComplexType<float>::type>& buffer,
    GPUArrayView2D<typename gpu::fft::ComplexType<float>::type> freqZData) -> void {
  compact_buffered_unpack_forward_launch(stream, maxNumXYPlanes, numXYPlanes, xyPlaneOffsets,
                                         buffer, freqZData);
}

auto compact_buffered_unpack_forward(
    const gpu::StreamType stream, const int maxNumXYPlanes, const GPUArrayView1D<int> numXYPlanes,
    const GPUArrayView1D<int>& xyPlaneOffsets,
    const GPUArrayView1D<typename gpu::fft::ComplexType<float>::type>& buffer,
    GPUArrayView2D<typename gpu::fft::ComplexType<double>::type> freqZData) -> void {
  compact_buffered_unpack_forward_launch(stream, maxNumXYPlanes, numXYPlanes, xyPlaneOffsets,
                                         buffer, freqZData);
}

template <typename DATA_TYPE, typename BUFFER_TYPE>
__global__ static void compact_buffered_pack_forward_kernel(
    const int maxNumZSticks, const GPUArrayConstView1D<int> numZSticks,
    const GPUArrayConstView1D<int> indices, const GPUArrayConstView2D<DATA_TYPE> freqXYData,
    GPUArrayView1D<BUFFER_TYPE> buffer) {
  const int xyPlaneIndex = threadIdx.x + blockIdx.x * blockDim.x;
  int bufferOffset = 0;
  if (xyPlaneIndex < freqXYData.dim_outer()) {
    for (int r = 0; r < numZSticks.size(); ++r) {
      const int numCurrentZSticks = numZSticks(r);
      for (int zStickIndex = blockIdx.y; zStickIndex < numCurrentZSticks;
           zStickIndex += gridDim.y) {
        const int currentIndex = indices(r * maxNumZSticks + zStickIndex);
        buffer(bufferOffset + zStickIndex * freqXYData.dim_outer() + xyPlaneIndex) =
            ConvertComplex<BUFFER_TYPE, DATA_TYPE>::apply(freqXYData(xyPlaneIndex, currentIndex));
      }
      bufferOffset += numCurrentZSticks * freqXYData.dim_outer();
    }
  }
}

template <typename DATA_TYPE, typename BUFFER_TYPE>
static auto compact_buffered_pack_forward_launch(const gpu::StreamType stream,
                                                 const int maxNumZSticks,
                                                 const GPUArrayView1D<int>& numZSticks,
                                                 const GPUArrayView1D<int>& indices,
                                                 const GPUArrayView3D<DATA_TYPE>& freqXYData,
                                                 GPUArrayView1D<BUFFER_TYPE> buffer) -> void {
  const dim3 threadBlock(gpu::BlockSizeSmall);
  const dim3 threadGrid((freqXYData.dim_outer() + threadBlock.x - 1) / threadBlock.x,
                        std::min(maxNumZSticks, gpu::GridSizeMedium));
  launch_kernel(compact_buffered_pack_forward_kernel<DATA_TYPE, BUFFER_TYPE>, threadGrid,
                threadBlock, 0, stream, maxNumZSticks, numZSticks, indices,
                GPUArrayConstView2D<DATA_TYPE>(freqXYData.data(), freqXYData.dim_outer(),
                                               freqXYData.dim_mid() * freqXYData.dim_inner(),
                                               freqXYData.device_id()),
                buffer);
}

auto compact_buffered_pack_forward(
    const gpu::StreamType stream, const int maxNumZSticks, const GPUArrayView1D<int>& numZSticks,
    const GPUArrayView1D<int>& indices,
    const GPUArrayView3D<typename gpu::fft::ComplexType<double>::type>& freqXYData,
    GPUArrayView1D<typename gpu::fft::ComplexType<double>::type> buffer) -> void {
  compact_buffered_pack_forward_launch(stream, maxNumZSticks, numZSticks, indices, freqXYData,
                                       buffer);
}

auto compact_buffered_pack_forward(
    const gpu::StreamType stream, const int maxNumZSticks, const GPUArrayView1D<int>& numZSticks,
    const GPUArrayView1D<int>& indices,
    const GPUArrayView3D<typename gpu::fft::ComplexType<float>::type>& freqXYData,
    GPUArrayView1D<typename gpu::fft::ComplexType<float>::type> buffer) -> void {
  compact_buffered_pack_forward_launch(stream, maxNumZSticks, numZSticks, indices, freqXYData,
                                       buffer);
}

auto compact_buffered_pack_forward(
    const gpu::StreamType stream, const int maxNumZSticks, const GPUArrayView1D<int>& numZSticks,
    const GPUArrayView1D<int>& indices,
    const GPUArrayView3D<typename gpu::fft::ComplexType<double>::type>& freqXYData,
    GPUArrayView1D<typename gpu::fft::ComplexType<float>::type> buffer) -> void {
  compact_buffered_pack_forward_launch(stream, maxNumZSticks, numZSticks, indices, freqXYData,
                                       buffer);
}

}  // namespace spfft
