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

#ifndef SPFFT_GPU_TRANSFER_HPP
#define SPFFT_GPU_TRANSFER_HPP

#include <cassert>
#include "gpu_util/gpu_stream_handle.hpp"
#include "memory/memory_type_trait.hpp"
#include "spfft/config.h"
#include "util/common_types.hpp"

namespace spfft {

template <typename T, typename U>
auto copy_to_gpu(const T& hostArray, U&& gpuArray) -> void {
  using UType = typename std::remove_reference<U>::type;
  static_assert(!IsDeviceMemory<T>::value, "First argument must represent host memory!");
  static_assert(IsDeviceMemory<UType>::value, "Second argument must represent device memory!");
  static_assert(sizeof(decltype(*(gpuArray.data()))) == sizeof(decltype(*(hostArray.data()))),
                "Size of value types must match!");

  assert(hostArray.size() == static_cast<SizeType>(gpuArray.size()));
  gpu::check_status(gpu::memcpy(
      static_cast<void*>(gpuArray.data()), static_cast<const void*>(hostArray.data()),
      gpuArray.size() * sizeof(decltype(*(gpuArray.data()))), gpu::flag::MemcpyHostToDevice));
}

template <typename T, typename U>
auto copy_to_gpu_async(const GPUStreamHandle& stream, const T& hostArray, U&& gpuArray) -> void {
  using UType = typename std::remove_reference<U>::type;
  static_assert(!IsDeviceMemory<T>::value, "First argument must represent host memory!");
  static_assert(IsDeviceMemory<UType>::value, "Second argument must represent device memory!");
  static_assert(sizeof(decltype(*(gpuArray.data()))) == sizeof(decltype(*(hostArray.data()))),
                "Size of value types must match!");

  assert(hostArray.size() == static_cast<SizeType>(gpuArray.size()));
  gpu::check_status(gpu::memcpy_async(static_cast<void*>(gpuArray.data()),
                                      static_cast<const void*>(hostArray.data()),
                                      gpuArray.size() * sizeof(decltype(*(gpuArray.data()))),
                                      gpu::flag::MemcpyHostToDevice, stream.get()));
}

template <typename T, typename U>
auto copy_from_gpu(const T& gpuArray, U&& hostArray) -> void {
  using UType = typename std::remove_reference<U>::type;
  static_assert(IsDeviceMemory<T>::value, "First argument must represent device memory!");
  static_assert(!IsDeviceMemory<UType>::value, "Second argument must represent host memory!");
  static_assert(sizeof(decltype(*(gpuArray.data()))) == sizeof(decltype(*(hostArray.data()))),
                "Size of value types must match!");

  assert(hostArray.size() == static_cast<SizeType>(gpuArray.size()));
  gpu::check_status(gpu::memcpy(
      static_cast<void*>(hostArray.data()), static_cast<const void*>(gpuArray.data()),
      hostArray.size() * sizeof(decltype(*(gpuArray.data()))), gpu::flag::MemcpyDeviceToHost));
}

template <typename T, typename U>
auto copy_from_gpu_async(const GPUStreamHandle& stream, const T& gpuArray, U&& hostArray) -> void {
  using UType = typename std::remove_reference<U>::type;
  static_assert(IsDeviceMemory<T>::value, "First argument must represent device memory!");
  static_assert(!IsDeviceMemory<UType>::value, "Second argument must represent host memory!");
  static_assert(sizeof(decltype(*(gpuArray.data()))) == sizeof(decltype(*(hostArray.data()))),
                "Size of value types must match!");

  assert(hostArray.size() == static_cast<SizeType>(gpuArray.size()));
  gpu::check_status(gpu::memcpy_async(static_cast<void*>(hostArray.data()),
                                      static_cast<const void*>(gpuArray.data()),
                                      hostArray.size() * sizeof(decltype(*(gpuArray.data()))),
                                      gpu::flag::MemcpyDeviceToHost, stream.get()));
}

}  // namespace spfft

#endif
