
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
#ifndef SPFFT_GPU_POINTER_TRANSLATION_HPP
#define SPFFT_GPU_POINTER_TRANSLATION_HPP

#include <gpu_util/gpu_runtime_api.hpp>
#include <utility>

#include "spfft/config.h"

namespace spfft {

// Translate input pointer to host / device pointer pair. Managed memory is not considered for
// device pointer.
template <typename T>
auto translate_gpu_pointer(const T* inputPointer) -> std::pair<const T*, const T*> {
  gpu::PointerAttributes attr;
  attr.devicePointer = nullptr;
  attr.hostPointer = nullptr;
  auto status = gpu::pointer_get_attributes(&attr, static_cast<const void*>(inputPointer));

  if (status != gpu::status::Success) {
    gpu::get_last_error();  // clear error from cache
    // Invalid value is always indicated before CUDA 11 for valid host pointers, which have not been
    // registered. -> Don't throw error in this case.
    if (status != gpu::status::ErrorInvalidValue) gpu::check_status(status);
  }

  std::pair<const T*, const T*> ptrPair{nullptr, nullptr};

  // get memory type - cuda 10 changed attribute name
#if defined(SPLA_CUDA) && (CUDART_VERSION >= 10000)
  auto memoryType = attr.type;
#else
  auto memoryType = attr.memoryType;
#endif

#if defined(SPLA_ROCM) && (HIP_VERSION < 310)
  // Workaround due to bug with HIP when parsing pointers with offset from allocated memory start.
  // Fixed in ROCm 3.10.
  if (memoryType != gpu::flag::MemoryTypeDevice) {
    ptrPair.first = inputPointer;
  } else {
    ptrPair.second = inputPointer;
  }

#else

  if (memoryType != gpu::flag::MemoryTypeDevice) {
    ptrPair.first = attr.hostPointer ? static_cast<const T*>(attr.hostPointer) : inputPointer;
  } else {
    ptrPair.second = static_cast<const T*>(attr.devicePointer);
  }
#endif

  return ptrPair;
}

template <typename T>
auto translate_gpu_pointer(T* inputPointer) -> std::pair<T*, T*> {
  auto pointers = translate_gpu_pointer(static_cast<const T*>(inputPointer));
  return {const_cast<T*>(pointers.first), const_cast<T*>(pointers.second)};
}

}  // namespace spfft

#endif
