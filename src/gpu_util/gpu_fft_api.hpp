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
#ifndef SPFFT_GPU_FFT_API_HPP
#define SPFFT_GPU_FFT_API_HPP

#include "spfft/config.h"

#if defined(SPFFT_CUDA)
#include <cufft.h>
#define GPU_FFT_PREFIX(val) cufft##val

#elif defined(SPFFT_ROCM)
#include <hipfft.h>
#define GPU_FFT_PREFIX(val) hipfft##val
#endif

// only declare namespace members if GPU support is enabled
#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)

#include <utility>
#include "spfft/exceptions.hpp"

namespace spfft {
namespace gpu {
namespace fft {

// ==================================
// Types
// ==================================
using ResultType = GPU_FFT_PREFIX(Result);
using HandleType = GPU_FFT_PREFIX(Handle);
using ComplexFloatType = GPU_FFT_PREFIX(Complex);
using ComplexDoubleType = GPU_FFT_PREFIX(DoubleComplex);

// Complex type selector
template <typename T>
struct ComplexType;

template <>
struct ComplexType<double> {
  using type = ComplexDoubleType;
};

template <>
struct ComplexType<float> {
  using type = ComplexFloatType;
};

// ==================================
// Transform types
// ==================================
namespace TransformDirection {
#ifdef SPFFT_CUDA
constexpr auto Forward = CUFFT_FORWARD;
constexpr auto Backward = CUFFT_INVERSE;
#else
constexpr auto Forward = HIPFFT_FORWARD;
constexpr auto Backward = HIPFFT_BACKWARD;
#endif
} // namespace TransformDirection

// ==================================
// Transform types
// ==================================
namespace TransformType {
#ifdef SPFFT_CUDA
constexpr auto R2C = CUFFT_R2C;
constexpr auto C2R = CUFFT_C2R;
constexpr auto C2C = CUFFT_C2C;
constexpr auto D2Z = CUFFT_D2Z;
constexpr auto Z2D = CUFFT_Z2D;
constexpr auto Z2Z = CUFFT_Z2Z;
#else
constexpr auto R2C = HIPFFT_R2C;
constexpr auto C2R = HIPFFT_C2R;
constexpr auto C2C = HIPFFT_C2C;
constexpr auto D2Z = HIPFFT_D2Z;
constexpr auto Z2D = HIPFFT_Z2D;
constexpr auto Z2Z = HIPFFT_Z2Z;
#endif

// Transform type selector
template <typename T>
struct ComplexToComplex;

template <>
struct ComplexToComplex<double> {
  constexpr static auto value = Z2Z;
};

template <>
struct ComplexToComplex<float> {
  constexpr static auto value = C2C;
};

// Transform type selector
template <typename T>
struct RealToComplex;

template <>
struct RealToComplex<double> {
  constexpr static auto value = D2Z;
};

template <>
struct RealToComplex<float> {
  constexpr static auto value = R2C;
};

// Transform type selector
template <typename T>
struct ComplexToReal;

template <>
struct ComplexToReal<double> {
  constexpr static auto value = Z2D;
};

template <>
struct ComplexToReal<float> {
  constexpr static auto value = C2R;
};
} // namespace TransformType

// ==================================
// Result values
// ==================================
namespace result {
#ifdef SPFFT_CUDA
constexpr auto Success = CUFFT_SUCCESS;
#else
constexpr auto Success = HIPFFT_SUCCESS;
#endif
} // namespace result

// ==================================
// Error check functions
// ==================================
inline auto check_result(ResultType error) -> void {
  if (error != result::Success) {
    throw GPUFFTError();
  }
}

// ==================================
// Execution function overload
// ==================================
inline auto execute(HandleType& plan, ComplexDoubleType* iData, double* oData) -> ResultType {
  return GPU_FFT_PREFIX(ExecZ2D)(plan, iData, oData);
}

inline auto execute(HandleType& plan, ComplexFloatType* iData, float* oData) -> ResultType {
  return GPU_FFT_PREFIX(ExecC2R)(plan, iData, oData);
}

inline auto execute(HandleType& plan, double* iData, ComplexDoubleType* oData) -> ResultType {
  return GPU_FFT_PREFIX(ExecD2Z)(plan, iData, oData);
}

inline auto execute(HandleType& plan, float* iData, ComplexFloatType* oData) -> ResultType {
  return GPU_FFT_PREFIX(ExecR2C)(plan, iData, oData);
}

inline auto execute(HandleType& plan, ComplexDoubleType* iData, ComplexDoubleType* oData,
                    int direction) -> ResultType {
  return GPU_FFT_PREFIX(ExecZ2Z)(plan, iData, oData, direction);
}

inline auto execute(HandleType& plan, ComplexFloatType* iData, ComplexFloatType* oData,
                    int direction) -> ResultType {
  return GPU_FFT_PREFIX(ExecC2C)(plan, iData, oData, direction);
}

// ==================================
// Forwarding functions of to GPU API
// ==================================
template <typename... ARGS>
inline auto create(ARGS... args) -> ResultType {
  return GPU_FFT_PREFIX(Create)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto make_plan_many(ARGS... args) -> ResultType {
  return GPU_FFT_PREFIX(MakePlanMany)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto set_work_area(ARGS... args) -> ResultType {
  return GPU_FFT_PREFIX(SetWorkArea)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto destroy(ARGS... args) -> ResultType {
  return GPU_FFT_PREFIX(Destroy)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto set_stream(ARGS... args) -> ResultType {
  return GPU_FFT_PREFIX(SetStream)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto set_auto_allocation(ARGS... args) -> ResultType {
  return GPU_FFT_PREFIX(SetAutoAllocation)(std::forward<ARGS>(args)...);
}

} // namespace fft
} // namespace gpu
} // namespace spfft

#undef GPU_FFT_PREFIX

#endif // defined SPFFT_CUDA || SPFFT_ROCM
#endif
