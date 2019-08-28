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
#ifndef SPFFT_GPU_RUNTIME_HPP
#define SPFFT_GPU_RUNTIME_HPP

#include "spfft/config.h"
#include "gpu_util/gpu_runtime_api.hpp"

#ifdef SPFFT_ROCM
#include <hip/hip_runtime.h>
#endif

namespace spfft {

#ifdef SPFFT_CUDA
template <typename F, typename... ARGS>
inline auto launch_kernel(F func, const dim3 threadGrid, const dim3 threadBlock,
                          const size_t sharedMemoryBytes, const gpu::StreamType stream,
                          ARGS... args) -> void {
#ifndef NDEBUG
  gpu::device_synchronize();
  gpu::check_status(gpu::get_last_error()); // before
#endif
  func<<<threadGrid, threadBlock,sharedMemoryBytes, stream>>>(std::forward<ARGS>(args)...);
#ifndef NDEBUG
  gpu::device_synchronize();
  gpu::check_status(gpu::get_last_error()); // after
#endif
}
#endif

#ifdef SPFFT_ROCM
template <typename F, typename... ARGS>
inline auto launch_kernel(F func, const dim3 threadGrid, const dim3 threadBlock,
                          const size_t sharedMemoryBytes, const gpu::StreamType stream,
                          ARGS... args) -> void {
#ifndef NDEBUG
  gpu::device_synchronize();
  gpu::check_status(gpu::get_last_error()); // before
#endif
  hipLaunchKernelGGL(func, threadGrid, threadBlock, sharedMemoryBytes, stream,
                     std::forward<ARGS>(args)...);
#ifndef NDEBUG
  gpu::device_synchronize();
  gpu::check_status(gpu::get_last_error()); // after
#endif
}
#endif


} // namespace spfft

#endif
