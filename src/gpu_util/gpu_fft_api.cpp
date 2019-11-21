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

#include "spfft/config.h"
#include "gpu_util/gpu_fft_api.hpp"
// only declare namespace members if GPU support is enabled
#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)

namespace spfft {
namespace gpu {
namespace fft {
namespace TransformType {

constexpr decltype(ComplexToComplex<double>::value) ComplexToComplex<double>::value;
constexpr decltype(ComplexToComplex<float>::value) ComplexToComplex<float>::value;

constexpr decltype(RealToComplex<double>::value) RealToComplex<double>::value;
constexpr decltype(RealToComplex<float>::value) RealToComplex<float>::value;

constexpr decltype(ComplexToReal<double>::value) ComplexToReal<double>::value;
constexpr decltype(ComplexToReal<float>::value) ComplexToReal<float>::value;

}  // namespace TransformType
}  // namespace fft
}  // namespace gpu
}  // namespace spfft

#endif
