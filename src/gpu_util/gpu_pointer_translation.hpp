
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

template <typename T>
auto translate_gpu_pointer(const T* inputPointer)
    -> std::pair<const T*, const T*> {
  gpu::PointerAttributes attr;
  auto status = gpu::pointer_get_attributes(&attr, static_cast<const void*>(inputPointer));
  gpu::get_last_error(); // if pointer is not registered, error will be stored and has to be cleared

#ifdef SPFFT_ROCM
  if(status != gpu::status::Success) {
#else
  if(status == gpu::status::ErrorInvalidValue) {
#endif
    // not registered with cuda -> host pointer
    const T* devicePtr = nullptr;
    return {inputPointer, devicePtr};
  } else {
    gpu::check_status(status);
    return {static_cast<const T*>(attr.hostPointer), static_cast<const T*>(attr.devicePointer)};
  }

}

template <typename T>
auto translate_gpu_pointer(T* inputPointer)
    -> std::pair<T*, T*> {
  gpu::PointerAttributes attr;
  auto status = gpu::pointer_get_attributes(&attr, static_cast<const void*>(inputPointer));
  gpu::get_last_error(); // if pointer is not registered, error will be stored and has to be cleared

      gpu::get_last_error();
#ifdef SPFFT_ROCM
  if(status != gpu::status::Success) {
#else
  if(status == gpu::status::ErrorInvalidValue) {
#endif
    // not registered with cuda -> host pointer
    T* devicePtr = nullptr;
    return {inputPointer, devicePtr};
  } else {
    gpu::check_status(status);
    return {static_cast<T*>(attr.hostPointer), static_cast<T*>(attr.devicePointer)};
  }

}

} // namespace spfft

#endif
