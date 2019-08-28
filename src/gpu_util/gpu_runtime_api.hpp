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
#ifndef SPFFT_GPU_RUNTIME_RUNTIME_HPP
#define SPFFT_GPU_RUNTIME_RUNTIME_HPP

#include "spfft/config.h"

#if defined(SPFFT_CUDA)
#include <cuda_runtime_api.h>
#define GPU_PREFIX(val) cuda##val

#elif defined(SPFFT_ROCM)
#include <hip/hip_runtime_api.h>
#define GPU_PREFIX(val) hip##val
#endif

// only declare namespace members if GPU support is enabled
#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)

#include <utility>
#include "spfft/exceptions.hpp"

namespace spfft {
namespace gpu {

using StatusType = GPU_PREFIX(Error_t);
using StreamType = GPU_PREFIX(Stream_t);

#ifdef SPFFT_CUDA
using PointerAttributes = GPU_PREFIX(PointerAttributes);
#else
using PointerAttributes = GPU_PREFIX(PointerAttribute_t);
#endif

namespace status {
// error / return values
constexpr StatusType Success = GPU_PREFIX(Success);
constexpr StatusType ErrorMemoryAllocation = GPU_PREFIX(ErrorMemoryAllocation);
constexpr StatusType ErrorLaunchOutOfResources = GPU_PREFIX(ErrorLaunchOutOfResources);
constexpr StatusType ErrorInvalidValue = GPU_PREFIX(ErrorInvalidValue);
constexpr StatusType ErrorInvalidResourceHandle = GPU_PREFIX(ErrorInvalidResourceHandle);
constexpr StatusType ErrorInvalidDevice = GPU_PREFIX(ErrorInvalidDevice);
constexpr StatusType ErrorInvalidMemcpyDirection = GPU_PREFIX(ErrorInvalidMemcpyDirection);
constexpr StatusType ErrorInvalidDevicePointer = GPU_PREFIX(ErrorInvalidDevicePointer);
constexpr StatusType ErrorInitializationError = GPU_PREFIX(ErrorInitializationError);
constexpr StatusType ErrorNoDevice = GPU_PREFIX(ErrorNoDevice);
constexpr StatusType ErrorNotReady = GPU_PREFIX(ErrorNotReady);
constexpr StatusType ErrorUnknown = GPU_PREFIX(ErrorUnknown);
constexpr StatusType ErrorPeerAccessNotEnabled = GPU_PREFIX(ErrorPeerAccessNotEnabled);
constexpr StatusType ErrorPeerAccessAlreadyEnabled = GPU_PREFIX(ErrorPeerAccessAlreadyEnabled);
constexpr StatusType ErrorHostMemoryAlreadyRegistered =
    GPU_PREFIX(ErrorHostMemoryAlreadyRegistered);
constexpr StatusType ErrorHostMemoryNotRegistered = GPU_PREFIX(ErrorHostMemoryNotRegistered);
constexpr StatusType ErrorUnsupportedLimit = GPU_PREFIX(ErrorUnsupportedLimit);
} // namespace status

// flags to pass to GPU API
namespace flag {
constexpr auto HostRegisterDefault = GPU_PREFIX(HostRegisterDefault);
constexpr auto HostRegisterPortable = GPU_PREFIX(HostRegisterPortable);
constexpr auto HostRegisterMapped = GPU_PREFIX(HostRegisterMapped);
constexpr auto HostRegisterIoMemory = GPU_PREFIX(HostRegisterIoMemory);

constexpr auto StreamDefault = GPU_PREFIX(StreamDefault);
constexpr auto StreamNonBlocking = GPU_PREFIX(StreamNonBlocking);

constexpr auto MemoryTypeHost = GPU_PREFIX(MemoryTypeHost);
constexpr auto MemoryTypeDevice = GPU_PREFIX(MemoryTypeDevice);
#if (CUDART_VERSION >= 10000)
constexpr auto MemoryTypeUnregistered = GPU_PREFIX(MemoryTypeUnregistered);
constexpr auto MemoryTypeManaged = GPU_PREFIX(MemoryTypeManaged);
#endif

constexpr auto MemcpyHostToDevice = GPU_PREFIX(MemcpyHostToDevice);
constexpr auto MemcpyDeviceToHost = GPU_PREFIX(MemcpyDeviceToHost);
} // namespace flag

// ==================================
// Error check functions
// ==================================
inline auto check_status(StatusType error) -> void {
  if (error != status::Success) {
    if (error == status::ErrorMemoryAllocation) throw GPUAllocationError();
    if (error == status::ErrorLaunchOutOfResources) throw GPULaunchError();
    if (error == status::ErrorNoDevice) throw GPUNoDeviceError();
    if (error == status::ErrorInvalidValue) throw GPUInvalidValueError();
    if (error == status::ErrorInvalidDevicePointer) throw GPUInvalidDevicePointerError();

    throw GPUError();
  }
}

// ==================================
// Forwarding functions of to GPU API
// ==================================
template <typename... ARGS>
inline auto host_register(ARGS... args) -> StatusType {
  return GPU_PREFIX(HostRegister)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto host_unregister(ARGS... args) -> StatusType {
  return GPU_PREFIX(HostUnregister)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto stream_create_with_flags(ARGS... args) -> StatusType {
  return GPU_PREFIX(StreamCreateWithFlags)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto stream_destroy(ARGS... args) -> StatusType {
  return GPU_PREFIX(StreamDestroy)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto malloc(ARGS... args) -> StatusType {
  return GPU_PREFIX(Malloc)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto free(ARGS... args) -> StatusType {
  return GPU_PREFIX(Free)(std::forward<ARGS>(args)...);
}


template <typename... ARGS>
inline auto memcpy(ARGS... args) -> StatusType {
  return GPU_PREFIX(Memcpy)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto memcpy_async(ARGS... args) -> StatusType {
  return GPU_PREFIX(MemcpyAsync)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto get_device(ARGS... args) -> StatusType {
  return GPU_PREFIX(GetDevice)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto set_device(ARGS... args) -> StatusType {
  return GPU_PREFIX(SetDevice)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto get_device_count(ARGS... args) -> StatusType {
  return GPU_PREFIX(GetDeviceCount)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto stream_synchronize(ARGS... args) -> StatusType {
  return GPU_PREFIX(StreamSynchronize)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto memset_async(ARGS... args) -> StatusType {
  return GPU_PREFIX(MemsetAsync)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto pointer_get_attributes(ARGS... args) -> StatusType {
  return GPU_PREFIX(PointerGetAttributes)(std::forward<ARGS>(args)...);
}

inline auto get_last_error() -> StatusType {
  return GPU_PREFIX(GetLastError)();
}

inline auto device_synchronize() -> StatusType {
  return GPU_PREFIX(DeviceSynchronize)();
}

} // namespace gpu
} // namespace spfft

#undef GPU_PREFIX

#endif // defined SPFFT_CUDA || SPFFT_ROCM
#endif
