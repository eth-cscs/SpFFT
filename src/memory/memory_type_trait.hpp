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
#ifndef SPFFT_MEMORY_TYPE_TRAIT_HPP
#define SPFFT_MEMORY_TYPE_TRAIT_HPP
#include "spfft/config.h"

#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
#include "memory/gpu_array.hpp"
#include "memory/gpu_array_view.hpp"
#include "memory/gpu_array_const_view.hpp"
#endif

namespace spfft {

template<typename T>
struct IsDeviceMemory {
  constexpr static bool value = false;
};

#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
template<typename T>
struct IsDeviceMemory<GPUArray<T>> {
  constexpr static bool value = true;
};

template<typename T>
struct IsDeviceMemory<GPUArrayView1D<T>> {
  constexpr static bool value = true;
};

template<typename T>
struct IsDeviceMemory<GPUArrayView2D<T>> {
  constexpr static bool value = true;
};
template<typename T>
struct IsDeviceMemory<GPUArrayView3D<T>> {
  constexpr static bool value = true;
};


template<typename T>
struct IsDeviceMemory<GPUArrayConstView1D<T>> {
  constexpr static bool value = true;
};

template<typename T>
struct IsDeviceMemory<GPUArrayConstView2D<T>> {
  constexpr static bool value = true;
};
template<typename T>
struct IsDeviceMemory<GPUArrayConstView3D<T>> {
  constexpr static bool value = true;
};

#endif
} // namespace
#endif
